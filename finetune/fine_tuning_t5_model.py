import json
# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import AdamW, T5Tokenizer, T5ForConditionalGeneration

from rich.table import Column, Table
from rich import box
from rich.console import Console


train_data_path = "/train_data.jsonl"
model_file_path = "/model_files"
output_path = "/out_model"


# TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(output_path + '/runs')


# Train Data ###############################################################################################################################

train_data = []
with open(train_data_path, 'r') as fin:
    for i in fin.readlines():
        train_data.append(json.loads(i))
train_data = train_data[0]


print('train_data : ', len(train_data))


for data in train_data:
  for i in data['articles']:
      if(len(i)==0):
          data['articles'].remove(i)


########################################################################################################################################

# define a rich console logger
console=Console(record=True)

def display_df(df):
  """display dataframe in ASCII format"""

  console=Console()
  table = Table(Column("source_text", justify="center" ), Column("target_text", justify="center"), title="Sample Data",pad_edge=False, box=box.ASCII)

  for i, row in enumerate(df.values.tolist()):
    table.add_row(row[0], row[1])

  console.print(table)


# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


class ArticleDataSetClass(Dataset):
  """
  Creating a custom dataset for reading the dataset and 
  loading it into the dataloader to pass it to the neural network for finetuning the model

  """

  def __init__(self, dataframe, tokenizer, source_len):
    self.tokenizer = tokenizer
    self.data = dataframe
    self.source_len = source_len
    self.source_text = self.data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index): # foreach dataset
    source_text = str(self.source_text[index])

    #cleaning data so as to ensure data is in string type
    source_text = ' '.join(source_text.split())

    source = self.tokenizer.batch_encode_plus([source_text], max_length= self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
  
    source_ids = source['input_ids'].squeeze()
    source_mask = source['attention_mask'].squeeze()

    return {
        'source_ids': source_ids.to(dtype=torch.long), 
        'source_mask': source_mask.to(dtype=torch.long)
    }

def generateArticle(epoch, tokenizer, model, device, loader):

  """
  Function to evaluate model for predictions
  """
  model.eval()
  with torch.no_grad():
    for _, data in enumerate(loader, 0):
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        generated_ids = model.generate(
            input_ids = ids,
            attention_mask = mask, 
            max_length=1024, 
            num_beams=2,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True
            )

        prediction = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        

  return prediction[0]

class TargetDataSetClass(Dataset):
  """
  Creating a custom dataset for reading the dataset and 
  loading it into the dataloader to pass it to the neural network for finetuning the model

  """

  def __init__(self, dataframe, tokenizer, source_len, target_len, source_text, target_text):
    self.tokenizer = tokenizer
    self.data = dataframe
    self.source_len = source_len
    self.summ_len = target_len
    print("self.data : ", self.data)
    self.source_text = [self.data[source_text]]
    self.target_text = [self.data[target_text]]

  def __len__(self):
    return len(self.target_text)

  def __getitem__(self, index):
    source_text = str(self.source_text[index])
    target_text = str(self.target_text[index])

    #cleaning data so as to ensure data is in string type
    source_text = ' '.join(source_text.split())
    target_text = ' '.join(target_text.split())

    source = self.tokenizer.batch_encode_plus([source_text], max_length= self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
    target = self.tokenizer.batch_encode_plus([target_text], max_length= self.summ_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

    source_ids = source['input_ids'].squeeze()
    source_mask = source['attention_mask'].squeeze()
    target_ids = target['input_ids'].squeeze()
    target_mask = target['attention_mask'].squeeze()

    return {
        'source_ids': source_ids.to(dtype=torch.long), 
        'source_mask': source_mask.to(dtype=torch.long), 
        'target_ids': target_ids.to(dtype=torch.long),
        'target_ids_y': target_ids.to(dtype=torch.long)
    }

def train(epoch, tokenizer, model, device, loader, optimizer):
  print("Training Model")
  print("len loader : ", len(loader))
  model.train()
  for _,data in enumerate(loader, 0):

    y = data['target_ids'].to(device, dtype = torch.long)
    y_ids = y[:, :-1].contiguous()
    lm_labels = y[:, 1:].clone().detach()
    lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
    ids = data['source_ids'].to(device, dtype = torch.long)
    mask = data['source_mask'].to(device, dtype = torch.long)

    outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
    loss = outputs[0]

    # TensorBoard 
    writer.add_scalar("Loss/train", loss, _ )

    console.print(str(epoch), str(_), str(loss))


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

"""#**Here !**"""

def generationAndTrainModel(epoch, tokenizer, model, device, traindata, optimizer, output_dir):

  """
  Function to be called for training with the parameters passed from main function
  """
  for traindataIndex, data in enumerate(traindata, 0):
    print('traindataIndex : ', traindataIndex)

    articleRecord = []
    articleRecord.append(data['first_summary'])

    articlesLast = data['articles'][-1]
    articles = data['articles'][:-1]

    for articleIndex, article in enumerate(articles, 0): 
      print('Index : ', articleIndex)   

      combineArticle = "update: " + articleRecord[articleIndex]+ " event: " + article

      # 生成 articleIndex 的 Article
      # Encode input
      article_set = ArticleDataSetClass([combineArticle], tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"])
      # DataLoader
      gen_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 4
        }
      # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
      gen_loader = DataLoader(article_set, **gen_params)
      # Decode output
      generateArticle_ = generateArticle(epoch, tokenizer, model, device, gen_loader)
      articleRecord.append(generateArticle_)

#-------------------------------------------------------------------------------------------------------------
    """
    Training Model

    """

    finalArticle_Source = "update: " + articleRecord[-1]+ " event: " + articlesLast
    final_Target = data['final_summary']
    train_Source_Target = {"source": finalArticle_Source, "target": final_Target}

    # Encode input
    training_set = TargetDataSetClass(train_Source_Target, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], "source", "target")

    # DataLoader
    train_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 4
        }
    # Creation of Dataloaders for training
    training_set_loader = DataLoader(training_set, **train_params)
    print(len(training_set_loader))

    # Training Model
    train(epoch, tokenizer, model, device, training_set_loader, optimizer)

    # save Model(thousand)
    dataset_thousand = traindataIndex % 500
    if dataset_thousand == 0:
      console.log(f"[Saving Model]...\n")
      #Saving the model after training
      path = os.path.join(output_dir, "model_files")
      model.save_pretrained(path)
      tokenizer.save_pretrained(path)
      console.save_text(os.path.join(output_dir,'logs.txt'))




def validate(epoch, tokenizer, model, device, loader):

  """
  Function to evaluate model for predictions
  """
  
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=1024, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True
              )
         
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
          if _%10==0:
              console.print(f'Completed {_}')

          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals

def T5Trainer(dataframe, model_params, output_dir):

  
  """
  T5 trainer

  """

  # Set random seeds and deterministic pytorch for reproducibility
  torch.manual_seed(model_params["SEED"]) # pytorch random seed
  np.random.seed(model_params["SEED"]) # numpy random seed
  torch.backends.cudnn.deterministic = True

  # logging
  console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

  # tokenzier for encoding the text
  tokenizer = T5Tokenizer.from_pretrained(model_file_path)

  # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
  # Further this model is sent to device (GPU/TPU) for using the hardware.
  model = T5ForConditionalGeneration.from_pretrained(model_file_path)
  model = model.to(device)
  
  # logging
  console.log(f"[Data]: Reading data...\n")

  optimizer = torch.optim.AdamW( model.parameters(), lr=model_params["LEARNING_RATE"])


#----------------------------------------------------------------------------------------------------------
  # Training loop
  console.log(f'[Initiating Fine Tuning]...\n')

  for epoch in range(model_params["TRAIN_EPOCHS"]):
      generationAndTrainModel(epoch, tokenizer, model, device, dataframe, optimizer, output_dir)
      
#----------------------------------------------------------------------------------------------------------

 # TensorBoard 
  writer.flush()
  writer.close()

  console.log(f"[Saving Model]...\n")
  #Saving the model after training
  path = os.path.join(output_dir, "model_files")
  model.save_pretrained(path)
  tokenizer.save_pretrained(path)

  console.save_text(os.path.join(output_dir,'logs.txt'))
  
  console.print(f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n""")
  console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")
  print('--------------------------------------------------------------------------------------------')



model_params={
    "MODEL":"t5-base",             # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE":1,#1          # training batch size
    "VALID_BATCH_SIZE":1,#1          # validation batch size
    "TRAIN_EPOCHS":1,#3,              # number of training epochs
    "VAL_EPOCHS":1,                # number of validation epochs
    "LEARNING_RATE":1e-4,          # learning rate
    "MAX_SOURCE_TEXT_LENGTH":1024,#512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH":1024,#50,   # max length of target text
    "SEED": 42                     # set seed for reproducibility 
}

T5Trainer(train_data, model_params=model_params, output_dir=output_path)

