
impl ModernBertAttention {
    fn load(vb: VarBuilder, config: &Config, rotary_emb: Arc<RotaryEmbedding>) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;

        let qkv = linear_no_bias(config.hidden_size, config.hidden_size * 3, vb.pp("Wqkv"))?;
        let proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("Wo"))?;

        Ok(Self {
            qkv,
            proj,
            num_attention_heads,
            attention_head_size,
            rotary_emb,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let xs = hidden_states.clone();
        let (b, seq_len, d) = xs.dims3()?;
        
        // QKVの計算
        let qkv = xs
            .apply(&self.qkv)?
            .reshape((
                b,
                seq_len,
                3,
                self.num_attention_heads,
                self.attention_head_size,
            ))?
            .permute((2, 0, 3, 1, 4))?;

        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;

        // Rotary Embeddingの適用
        let (q, k) = self.rotary_emb.apply_rotary_emb_qkv(&q, &k)?;

        // Flash Attentionの使用
        let xs = self.forward_flash_attention(&q, &k, &v, attention_mask, b, seq_len, d)?;

        Ok(xs)
    }

    fn forward_flash_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: &Tensor,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<Tensor> {
        // Flash Attentionのための形状変換
        // (batch, num_heads, seq_len, head_dim) -> (total_tokens, num_heads, head_dim)
        let total_tokens = batch_size * seq_len;
        
        let q_flash = q.reshape((total_tokens, self.num_attention_heads, self.attention_head_size))?;
        let k_flash = k.reshape((total_tokens, self.num_attention_heads, self.attention_head_size))?;
        let v_flash = v.reshape((total_tokens, self.num_attention_heads, self.attention_head_size))?;

        // シーケンス長情報の準備
        // 各バッチのシーケンス開始位置を計算
        let mut seqlens_data = Vec::new();
        for i in 0..=batch_size {
            seqlens_data.push((i * seq_len) as u32);
        }
        let seqlens_q = Tensor::from_vec(seqlens_data.clone(), &[batch_size + 1], q.device())?;
        let seqlens_k = seqlens_q.clone();

        // スケール計算
        let softmax_scale = (self.attention_head_size as f64).powf(-0.5) as f32;

        // Flash Attentionの実行
        let attn_output = flash_attn_varlen_windowed(
            &q_flash,
            &k_flash,
            &v_flash,
            &seqlens_q,
            &seqlens_k,
            seq_len,      // max_seqlen_q
            seq_len,      // max_seqlen_k
            softmax_scale,
            None,         // window_size_left (無制限)
            None,         // window_size_right (無制限)
        )?;

        // 元の形状に戻す
        let xs = attn_output
            .reshape((batch_size, seq_len, self.num_attention_heads, self.attention_head_size))?
            .permute((0, 2, 1, 3))?  // (batch, num_heads, seq_len, head_dim)
            .reshape((batch_size, seq_len, hidden_size))?;

        // 出力プロジェクション
        let xs = xs.apply(&self.proj)?;

        Ok(xs)
    }

    // もしくは、よりシンプルなFlash Attention実装
    fn forward_simple_flash_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<Tensor> {
        // Flash Attentionライブラリが提供する標準的なインターフェースを使用
        let scale = (self.attention_head_size as f64).powf(-0.5) as f32;
        
        // Flash Attentionの実行（packed QKV形式を使用する場合）
        let qkv_packed = Tensor::cat(&[q, k, v], 2)?  // head次元でconcat
            .reshape((batch_size, seq_len, 3, self.num_attention_heads, self.attention_head_size))?;
        
        // Flash Attention関数（実際のAPIに応じて調整が必要）
        let attn_output = flash_attention_packed(
            &qkv_packed,
            scale,
            None,  // dropout_p
            None,  // causal
            None,  // window_size
        )?;

        // 形状を戻す
        let xs = attn_output
            .reshape((batch_size, seq_len, hidden_size))?;

        // 出力プロジェクション
        let xs = xs.apply(&self.proj)?;

        Ok(xs)
    }
}
