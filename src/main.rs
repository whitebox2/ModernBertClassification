use anyhow::{Error, Ok};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{VarBuilder, loss, ops, optim::Optimizer, var_map::VarMap};
use candle_optimisers::adam::{Adam, ParamsAdam};
use candle_transformers::models::modernbert::*;
use hf_hub::{Repo, RepoType, api::sync::Api};
use polars::prelude::*;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

const LR: f64 = 2e-5;
const EPOCHS: usize = 3;
const SEQ: usize = 20;

fn main() -> Result<(), Error> {
    let device = Device::new_metal(0)?;
    let repo = Api::new()?.repo(Repo::with_revision(
        "answerdotai/ModernBERT-base".to_string(),
        RepoType::Model,
        "main".into(),
    ));

    let mut config: Config =
        serde_json::from_str(&std::fs::read_to_string(repo.get("config.json")?)?)?;
    let mut tokenizer = Tokenizer::from_file(repo.get("tokenizer.json")?).map_err(Error::msg)?;

    tokenizer
        .with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(SEQ),
            pad_id: config.pad_token_id,
            pad_token: "[PAD]".into(),
            ..Default::default()
        }))
        .with_truncation(Some(TruncationParams {
            max_length: SEQ,
            ..Default::default()
        }))
        .map_err(Error::msg)?;

    let labels = ["Technology", "News", "Health", "Entertainment"];

    config.classifier_config = Some(ClassifierConfig {
        id2label: labels
            .iter()
            .enumerate()
            .map(|(i, l)| (i.to_string(), l.to_string()))
            .collect(),
        label2id: labels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.to_string(), i.to_string()))
            .collect(),
        classifier_pooling: ClassifierPooling::CLS,
    });

    let mut varmap = VarMap::new();
    varmap.load(repo.get("model.safetensors")?)?;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = ModernBertForSequenceClassification::load(vb.clone(), &config)?;

    let df_tokenized = get_train_data(tokenizer)?;
    let sample = prepare_batch_tensors(df_tokenized, &device)?;
    train_model(model, varmap, sample)?;

    Ok(())
}

fn train_model(
    model: ModernBertForSequenceClassification,
    varmap: VarMap,
    sample: (Vec<Tensor>, Vec<Tensor>, Vec<Tensor>),
) -> Result<(), Error> {
    for epoch in 0..EPOCHS {
        println!("Epoch {}/{}", epoch + 1, EPOCHS);
        let mut total_loss = 0.0;
        let (ref input_ids, ref attention_masks, ref labels) = sample;
        let len = input_ids.len();

        for i in 0..len {
            let input_id = &input_ids[i];
            let attention_mask = &attention_masks[i];
            let label = &labels[i];

            let mut opt = Adam::new(
                varmap.all_vars(),
                ParamsAdam {
                    lr: LR,
                    ..Default::default()
                },
            )?;

            let logits = model.forward(input_id, attention_mask)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss_value = loss::nll(&log_sm, label)?;
            opt.backward_step(&loss_value)?;

            let loss_scalar = loss_value.to_vec0::<f32>()?;
            total_loss += loss_scalar;

            if (i + 1) % 10 == 0 || i == input_ids.len() - 1 {
                println!(
                    "  Sample {}/{}, Loss: {:.4}",
                    i + 1,
                    input_ids.len(),
                    loss_scalar
                );
            }
        }

        println!("  Average Loss: {:.4}", total_loss / input_ids.len() as f32);
        println!();
    }

    Ok(())
}

fn get_train_data(tokenizer: Tokenizer) -> Result<DataFrame, Error> {
    let labels = Series::new(
        "label".into(),
        &["Technology", "News", "Health", "Entertainment"],
    );
    let texts = Series::new(
        "text".into(),
        &[
            "The new smartphone features a foldable display and 5G support.",
            "The government announced new economic policies today.",
            "Regular exercise and a balanced diet are key to staying healthy.",
            "The latest action movie broke box office records this weekend.",
        ],
    );

    let df = DataFrame::new(vec![labels.into_column(), texts.into_column()])?;

    let tokenized_data = df
        .column("text")?
        .str()?
        .into_no_null_iter()
        .map(|text| {
            let encoding = tokenizer.encode(text, true).expect("valid text");
            (
                encoding.get_ids().to_vec(),
                encoding.get_attention_mask().to_vec(),
            )
        })
        .collect::<Vec<_>>();

    let input_ids: Vec<Vec<u32>> = tokenized_data.iter().map(|(ids, _)| ids.clone()).collect();

    let attention_masks: Vec<Vec<u32>> = tokenized_data
        .iter()
        .map(|(_, mask)| mask.clone())
        .collect();

    let mut input_ids_builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
        "input_ids".into(),
        input_ids.len(),
        input_ids.iter().map(|v| v.len()).sum::<usize>(),
        DataType::UInt32,
    );
    for data in input_ids {
        input_ids_builder.append_slice(&data);
    }
    let input_ids_chunk = input_ids_builder.finish();
    let input_ids_column = Column::new("input_ids".into(), input_ids_chunk);

    let mut attention_mask_builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
        "attention_mask".into(),
        attention_masks.len(),
        attention_masks.iter().map(|v| v.len()).sum::<usize>(),
        DataType::UInt32,
    );
    for data in attention_masks {
        attention_mask_builder.append_slice(&data);
    }
    let attention_mask_chunk = attention_mask_builder.finish();
    let attention_mask_column = Column::new("attention_mask".into(), attention_mask_chunk);

    let df_tokenized = df.hstack(&[input_ids_column, attention_mask_column])?;
    println!("{:?}", df_tokenized);

    let master_df = df! {
        "category" => ["Technology", "News", "Health", "Entertainment"],
        "value" => [1, 2, 3,4],
    }?;
    let master_df = master_df
        .clone()
        .lazy()
        .with_column(col("value").cast(DataType::UInt32))
        .collect()?;

    let df_result: DataFrame = df_tokenized
        .clone()
        .lazy()
        .join(
            master_df.clone().lazy(),
            [col("label")],
            [col("category")],
            JoinArgs::new(JoinType::Left),
        )
        .with_columns([col("value").alias("category_encoded")])
        .drop(["value"])
        .collect()?;

    Ok(df_result)
}

fn prepare_batch_tensors(
    batch_df: DataFrame,
    device: &Device,
) -> Result<(Vec<Tensor>, Vec<Tensor>, Vec<Tensor>), Error> {
    let input_ids_series: Series = batch_df
        .column("input_ids")?
        .clone()
        .take_materialized_series();

    let attention_mask_series: Series = batch_df
        .column("attention_mask")?
        .clone()
        .take_materialized_series();

    let label_series: Series = batch_df
        .column("category_encoded")?
        .clone()
        .take_materialized_series();

    let input_ids_tensor = create_batch_tensors_from_series(input_ids_series, 2, device)?;
    let attention_mask_tensor = create_batch_tensors_from_series(attention_mask_series, 2, device)?;
    let labels_tensor = create_batch_tensors_from_series_label(label_series, 2, device)?;
    let result = (input_ids_tensor, attention_mask_tensor, labels_tensor);

    Ok(result)
}

fn create_batch_tensors_from_series(
    series: Series,
    batch_size: usize,
    device: &Device,
) -> Result<Vec<Tensor>, Error> {
    let row = series.list().expect("Expected list type");
    let int_vec: Vec<_> = row.into_no_null_iter().collect();

    let u32_vec: Vec<Vec<u32>> = int_vec
        .into_iter()
        .map(|x| {
            x.u32()
                .expect("Expected u32 type")
                .into_no_null_iter()
                .collect()
        })
        .collect();

    let mut batch_tensors = Vec::new();

    for chunk in u32_vec.chunks(batch_size) {
        let tensors_in_batch: Result<Vec<Tensor>, _> = chunk
            .iter()
            .map(|vec| {
                let input: Vec<u32> = vec.iter().map(|&x| x as u32).collect();
                Tensor::from_slice(&input, (input.len(),), device)
            })
            .collect();

        let tensors_in_batch = tensors_in_batch?;
        let batched_tensor = Tensor::stack(&tensors_in_batch, 0)?;
        batch_tensors.push(batched_tensor);
    }

    Ok(batch_tensors)
}

pub fn create_batch_tensors_from_series_label(
    series: Series,
    batch_size: usize,
    device: &Device,
) -> Result<Vec<Tensor>, Error> {
    let chunked = series.u32().expect("Expected Series of type u32");
    let values: Vec<u32> = chunked.into_no_null_iter().collect();

    let mut batch_tensors = Vec::new();

    for batch in values.chunks(batch_size) {
        let tensor = Tensor::new(batch, device)?;
        batch_tensors.push(tensor);
    }

    Ok(batch_tensors)
}
