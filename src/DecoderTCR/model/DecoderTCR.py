import torch
import lightning as pl
from DecoderTCR.utils.model_zoo import get_base_model


class DecoderTCR(torch.nn.Module):
    def __init__(self, base_name):
        super(DecoderTCR, self).__init__()
        self.model, _ = get_base_model(base_name)

    def forward(self, data, repr_layers=None, return_contacts=False):
        out = self.model(data, repr_layers=repr_layers, return_contacts=return_contacts)
        return out
    
    def get_contact(self, data):
        out = self.model(data, return_contacts=True, repr_layers=[33])
        return out
    
    def get_embeddings(self, data, layer=33, mean_pool=True):
        """
        Extract embeddings from the model.
        
        Args:
            data: Tokenized input sequences (batch_size, seq_len)
            layer: Layer number to extract embeddings from (default: 33)
            mean_pool: If True, return mean-pooled embeddings per sequence.
                      If False, return per-residue embeddings.
        
        Returns:
            embeddings: Tensor of shape (batch_size, seq_len, embed_dim) if mean_pool=False
                       or (batch_size, embed_dim) if mean_pool=True
        """
        out = self.model(data, repr_layers=[layer], return_contacts=False)
        embeddings = out['representations'][layer]
        if mean_pool:
            # Mean pool over sequence length (excluding special tokens)
            from DecoderTCR.utils import pad_idx
            sequence_embeddings = []
            for i in range(embeddings.shape[0]):
                seq_len = (data[i] != pad_idx).sum()
                # Skip CLS token at position 0 and EOS at the end
                seq_emb = embeddings[i, 1:seq_len-1].mean(dim=0)
                sequence_embeddings.append(seq_emb)
            embeddings = torch.stack(sequence_embeddings)
        else:
            from DecoderTCR.utils import pad_idx
            sequence_embeddings = []
            for i in range(embeddings.shape[0]):
                seq_len = (data[i] != pad_idx).sum()
                # Skip CLS token at position 0 and EOS at the end
                seq_emb = embeddings[i, 1:seq_len-1]
                sequence_embeddings.append(seq_emb)
            embeddings = torch.stack(sequence_embeddings)
        
        return embeddings


class DecoderTCRModel(pl.LightningModule):
    def __init__(self, base_model, learning_rate=None, betas='1e-5', eps=None, weight_decay=None, pretrained_checkpoint=None):
        super(DecoderTCRModel, self).__init__()
         
        self.model = DecoderTCR(base_name=base_model)
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.learning_rate = learning_rate
        self.betas = eval(betas)
        self.eps = eps
        self.weight_decay = weight_decay
        self.NLL = torch.nn.NLLLoss(ignore_index=-100)
        self.m = torch.nn.LogSoftmax(dim=1)

        self.ema_loss = None
        self.ema_decay = 0.99

        self.train_loss_sum = 0.0
        self.train_loss_count = 0

    def on_train_epoch_start(self):
        # Reset the moving average tracking variables at the start of each epoch
        self.train_loss_sum = 0.0
        self.train_loss_count = 0

    def forward(self, data, repr_layers=None, return_contacts=False):
        return self.model(data, repr_layers=repr_layers, return_contacts=return_contacts)

    def training_step(self, batch):
        out = self.model(batch['masked_input'].cuda())
        logits = out['logits'].view(-1, out['logits'].size()[-1])
        target = batch['masked_target'].view(-1).cuda()
        loss = self.loss_fct(logits, target)

        if self.ema_loss is None:
            self.ema_loss = loss.item()
        else:
            self.ema_loss = self.ema_decay * self.ema_loss + (1 - self.ema_decay) * loss.item()

        self.log('train_loss', loss, sync_dist=True)
        self.log('train_ema_loss', self.ema_loss, sync_dist=True)    
        return loss 

    def validation_step(self, batch):
        out = self.model(batch['masked_input'].cuda(1))
        logits = out['logits'].view(-1, out['logits'].size()[-1])
        target = batch['masked_target'].view(-1).cuda(1)
        
        loss = self.loss_fct(logits, target)
        nll = self.NLL(self.m(logits), target)
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        self.log('NLL', nll, on_epoch=True, sync_dist=True)
        self.log('Perplexity', torch.exp(nll), on_epoch=True, sync_dist=True)
        return loss, torch.exp(nll).item()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        return optimizer
    
    def get_contact(self, data):
        out = self.model.get_contact(data)
        return out
    
    def get_embeddings(self, data, layer=33, mean_pool=True):
        """
        Extract embeddings from the model.
        
        Args:
            data: Tokenized input sequences (batch_size, seq_len)
            layer: Layer number to extract embeddings from (default: 33)
            mean_pool: If True, return mean-pooled embeddings per sequence.
                      If False, return per-residue embeddings.
        
        Returns:
            embeddings: Tensor of shape (batch_size, seq_len, embed_dim) if mean_pool=False
                       or (batch_size, embed_dim) if mean_pool=True
        """
        return self.model.get_embeddings(data, layer=layer, mean_pool=mean_pool)