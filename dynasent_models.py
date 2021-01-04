import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers import RobertaTokenizer


__author__ = 'Christopher Potts'


class DynaSentModel:
    def __init__(self, model_path, num_labels=3, max_length=128, device=None):
        self.model_path = model_path
        self.num_labels = num_labels
        self.max_length = max_length
        config = AutoConfig.from_pretrained(
            "roberta-base",
            num_labels=self.num_labels,
            finetuning_task='sst3')
        self.tokenizer = RobertaTokenizer.from_pretrained(
            'roberta-base',
            truncation=True,
            max_length=self.max_length,
            padding='max_length')
        self.classes = ['negative', 'positive', 'neutral']
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device
        self.device = torch.device(device)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            config=config)
        self.model.to(self.device)

    def predict_proba(self, strings):
        self.model.eval()
        with torch.no_grad():
            batch_encoding = self.tokenizer.batch_encode_plus(
                strings,
                max_length=self.max_length,
                padding='max_length',
                truncation=True)
            input_ids = torch.LongTensor(batch_encoding['input_ids'])
            input_ids = input_ids.to(self.device)
            attention_mask = torch.LongTensor(batch_encoding['attention_mask'])
            attention_mask = attention_mask.to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask)
            preds = F.softmax(outputs[0], dim=1)
            return preds.cpu().numpy()

    def predict(self, strings):
        preds = self.predict_proba(strings)
        return [self.classes[i] for i in preds.argmax(1)]


if __name__ == '__main__':
    model = DynaSentModel('models/dynasent_model0.bin')

    examples = [
        "superb",
        "They said the experience would be amazing, and they were right!",
        "They said the experience would be amazing, and they were wrong!"]

    cats = model.predict(examples)

    probs = model.predict_proba(examples)

    for ex, c, p in zip(examples, cats, probs):
        dist = dict(zip(model.classes, p))
        print(f"{ex}\n\t{c} {dist}\n")
