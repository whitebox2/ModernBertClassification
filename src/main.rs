use anyhow::Error;
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{VarBuilder, loss, ops, optim::Optimizer, var_map::VarMap};
use candle_optimisers::adam::{Adam, ParamsAdam};
use candle_transformers::models::modernbert::*;
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Encoding;

// use polars::prelude::*;

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

    let samples = get_train_data(&tokenizer, &device)?;

    train_model(model, varmap, samples)?;
    Ok(())
}

fn train_model(
    model: ModernBertForSequenceClassification,
    varmap: VarMap,
    samples: Vec<(Tensor, Tensor, Tensor)>,
) -> Result<(), Error> {
    for epoch in 0..EPOCHS {
        println!("Epoch {}/{}", epoch + 1, EPOCHS);
        let mut total_loss = 0.0;

        for (i, (input_id, attention_mask, label)) in samples.iter().enumerate() {
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

            if (i + 1) % 10 == 0 || i == samples.len() - 1 {
                println!(
                    "  Sample {}/{}, Loss: {:.4}",
                    i + 1,
                    samples.len(),
                    loss_scalar
                );
            }
        }

        println!("  Average Loss: {:.4}", total_loss / samples.len() as f32);
        println!();
    }

    Ok(())
}

fn get_train_data(
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<Vec<(Tensor, Tensor, Tensor)>, Error> {
    let sentences: Vec<&str> = vec![
        "The new smartphone features a foldable display and 5G support.",
        "The government announced new economic policies today.",
        "Regular exercise and a balanced diet are key to staying healthy.",
        "The latest action movie broke box office records this weekend.",
    ];

    let labels: Vec<u32> = vec![1, 2, 3, 4];

    let mut features: Vec<(Tensor, Tensor, Tensor)> = Vec::with_capacity(sentences.len());

    for (idx, text) in sentences.iter().enumerate() {
        let encoding: Encoding = tokenizer.encode(*text, true).expect("valid text");

        // input_ids（u32）→ Vec<u32> → Tensor
        let input_ids = encoding.get_ids();
        let tensor_input_ids = Tensor::new(input_ids, device)?;

        // attention_mask（u32）→ Vec<u32> → Tensor
        let mask = encoding.get_attention_mask();
        let tensor_mask = Tensor::new(mask, device)?;

        let label_value = labels[idx];
        let tensor_label = Tensor::new(label_value, device)?;

        features.push((tensor_input_ids, tensor_mask, tensor_label));
    }

    Ok(features)
}
