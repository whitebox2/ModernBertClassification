use anyhow::Error;
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{VarBuilder, loss::cross_entropy, ops, optim::Optimizer, var_map::VarMap};
use candle_optimisers::adam::{Adam, ParamsAdam};
use candle_transformers::models::modernbert::*;
use hf_hub::{Repo, RepoType, api::sync::Api};

use tokenizers::{PaddingParams, Tokenizer, TruncationParams};
const LR: f64 = 2e-5;
const EPOCHS: usize = 3;

const SEQ: usize = 70;

fn main() -> Result<(), Error> {
    let device = Device::new_metal(0)?;
    let repo = Api::new()?.repo(Repo::with_revision(
        "sbintuitions/modernbert-ja-30m".to_string(),
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

    let labels = ["News", "Entertainment", "Sports", "Technology"];

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

    let samples = get_train_data_batch(&tokenizer, &device, 2usize)?;
    println!("{:?}", samples);
    train_model_batch(model, varmap, samples)?;
    Ok(())
}

// training loop
fn train_model_batch(
    model: ModernBertForSequenceClassification,
    varmap: VarMap,
    batches: Vec<(Tensor, Tensor, Tensor)>,
) -> Result<(), Error> {
    for epoch in 0..EPOCHS {
        println!("Epoch {}/{}", epoch + 1, EPOCHS);
        let mut total_loss = 0.0;
        let mut total_samples = 0;

        for (i, (input_ids, attention_mask, labels)) in batches.iter().enumerate() {
            let mut opt = Adam::new(
                varmap.all_vars(),
                ParamsAdam {
                    lr: LR,
                    ..Default::default()
                },
            )?;

            let batch_size = input_ids.dim(0)?;

            let logits = model.forward(input_ids, attention_mask)?; // [batch_size, num_classes]

            println!("{:?}", logits);
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss_value = cross_entropy(&log_sm, labels)?;

            opt.backward_step(&loss_value)?;

            let loss_scalar = loss_value.to_vec0::<f32>()?;
            total_loss += loss_scalar * batch_size as f32;
            total_samples += batch_size;

            println!(
                "  Batch {}/{}, Loss: {:.4}, Batch Size: {}",
                i + 1,
                batches.len(),
                loss_scalar,
                batch_size
            );
        }

        println!("  Average Loss: {:.4}", total_loss / total_samples as f32);
        println!();
    }

    Ok(())
}

fn get_train_data_batch(
    tokenizer: &Tokenizer,
    device: &Device,
    batch_size: usize,
) -> Result<Vec<(Tensor, Tensor, Tensor)>, Error> {
    let sentences: Vec<&str> = vec![
        "The new smartphone features a foldable display and 5G support.",
        "The government announced new economic policies today.",
        "Regular exercise and a balanced diet are key to staying healthy.",
        "The latest action movie broke box office records this weekend.",
    ];
    let labels: Vec<u32> = vec![0, 1, 0];

    let mut batches = Vec::new();

    // batch
    for chunk in sentences.chunks(batch_size) {
        let chunk_labels: Vec<u32> = labels
            .iter()
            .skip(batches.len() * batch_size)
            .take(chunk.len())
            .cloned()
            .collect();

        let encodings: Vec<_> = chunk
            .iter()
            .map(|text| tokenizer.encode(*text, true).expect("valid text"))
            .collect();

        let batch_input_ids: Vec<Vec<u32>> =
            encodings.iter().map(|enc| enc.get_ids().to_vec()).collect();

        let batch_attention_mask: Vec<Vec<u32>> = encodings
            .iter()
            .map(|enc| enc.get_attention_mask().to_vec())
            .collect();

        let tensor_input_ids = Tensor::new(batch_input_ids, device)?; // [batch_size, seq_len]
        let tensor_mask = Tensor::new(batch_attention_mask, device)?; // [batch_size, seq_len]
        let tensor_labels = Tensor::new(chunk_labels, device)?; // [batch_size]

        batches.push((tensor_input_ids, tensor_mask, tensor_labels));
    }

    Ok(batches)
}

