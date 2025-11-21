import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from src.attention_layers import StandardAttention, MHA, MQA, GQA
from src.encoder import EncoderLayer
from src.decoder import DecoderLayer
from src.positional_encoding import FAPE


DEVICE = t.device('cuda' if t.cuda.is_available() else 'cpu')


class Transformer(nn.Module):
    def __init__(
            self,
            attention_module: StandardAttention | MHA | MQA | GQA,
            d_model: int,
            max_seq_len: int,
            tokenizer,
            device: t.device = DEVICE,
        ):
        super().__init__()
        
        # device
        self.device = device

        # dim, model name
        self.d_model = d_model
        self.attention_module = attention_module
        self.model_name = self.attention_module.__class__.__name__

        # tokenizer
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        # self.bos_token_id = self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else self.eos_token_id
        self.bos_token_id = self.eos_token_id
        self.max_seq_len = max_seq_len

        # encoder
        self.input_token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.positional_encoder = FAPE(d_model=self.d_model, max_seq_len=max_seq_len)
        self.dropout_encoder = nn.Dropout(0.1) # after embeddings
        self.encoder = EncoderLayer(attention_module=attention_module, d_model=d_model)
        
        # decoder 
        self.target_token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.dropout_decoder = nn.Dropout(0.1)
        self.decoder = DecoderLayer(attention_module=attention_module, d_model=d_model)
        
        # final linear/softmax head for logits
        self.linear_head = nn.Linear(d_model, self.vocab_size)


    def forward(self, input_ids: t.Tensor, Y_true: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        ## encoder
        # token embedding, scaling
        input_tokens = self.input_token_embedding(input_ids) * math.sqrt(self.d_model)
        # add positional encodings w/ dropout
        input_tokens = self.dropout_encoder(self.positional_encoder(input_tokens)) 
        # forward pass
        output_encoder = self.encoder(input_tokens)

        ## decoder
        # decoder input - Y_true shifted right to exclude the last item
        tgt_input_tokens = Y_true[:, :-1]
        # token embedding, scaling
        tgt = self.target_token_embedding(tgt_input_tokens) * math.sqrt(self.d_model)
        # add positional encodings w/ dropout
        tgt = self.dropout_decoder(self.positional_encoder(tgt)) 
        # forward pass
        output_decoder = self.decoder(tgt, output_encoder) 

        ## final output
        # linear head to project D_MODEL to vocab size
        logits = self.linear_head(output_decoder)
        # target labels - Y_true shifted left (T1, T2, ..., Tn, EOS)
        target_labels = Y_true[:, 1:] 
        return logits, target_labels
    
    
    # performs autoregressive decoding (inf) to generate target sequence
    def generate(self, input_ids: t.Tensor, max_length: int | None = None) -> t.Tensor:
        self.eval()
        B = input_ids.shape[0]
        max_length = max_length if max_length is not None else self.max_seq_len

        with t.no_grad():
            input_tokens = self.input_token_embedding(input_ids) * math.sqrt(self.d_model)
            input_tokens = self.dropout_encoder(self.positional_encoder(input_tokens)) 
            encoder_output = self.encoder(input_tokens)

            # prepare the initial decoder input (bos token)
            tgt_output = t.ones((B, 1), dtype=t.long, device=self.device) * self.bos_token_id
            is_finished = t.zeros(B, dtype=t.bool, device=self.device)

            # autoregressive decoding loop
            for _ in range(max_length - 1): # max length -  N-1 steps after bos
                tgt_emb = self.target_token_embedding(tgt_output) * math.sqrt(self.d_model) # decoder input = tgt_output
                tgt_emb = self.positional_encoder(tgt_emb)
                decoder_output = self.decoder(tgt_emb, encoder_output) 
                logits = self.linear_head(decoder_output[:, -1, :]) 
                
                # get the predicted token (greedy decoding - max probability)
                next_token_id = t.argmax(logits, dim=-1) # shape (B,)

                # update the status (finish if a sequence hits eos)
                is_finished = is_finished | (next_token_id == self.eos_token_id)
                
                # append the predicted token to the output sequence
                tgt_output = t.cat([tgt_output, next_token_id.unsqueeze(1)], dim=-1)
                
                # stop if all sequences completed
                if is_finished.all(): break
        
        # return the generated sequence including the bos and eos tokens
        return tgt_output

