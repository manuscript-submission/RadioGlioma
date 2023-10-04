# RadioGlioma
Repository linked to the research letter "Open-source Generative Artificial Intelligence for Information Extraction from Radiology Reports" submitted to Radiology.


## Installation

(Optional) - Download text-generation-webui https://github.com/oobabooga/text-generation-webui

1 - Download the vicuna 1.5 13B weights
If you installed text-generation-webui, just execute :

```bash
  python download-model.py lmsys/vicuna-13b-v1.5-16k
```
2- Install FastChat :

```bash
    pip3 install fschat
```
3- Edit the script_extraction.py file from this repository to update the paths to your Vicuna weights and source reports and name the table the script will create

4- Copy the auto_inference.py file from this repository to /FastChat/FastChat/serve

5- Edit the auto_inference.py file to match the variables you want to extract from the reports

6- Launch the script !

```bash
    python script_extraction
```


## Reports preparation

Reports must be in a singe .txt file, separated by a keyword you can change in auto_inference.py (default keyword is "NEXT_CASE")

## Prompting

You can modify the prompts from the script_extraction.py file.  
We advise you to use short, natural prompts.  
We find that asking for a list variables tagged with the caracteristics of interest is a good default prompt.  
Examples from the manuscript :  

## Available models

Nothing limits this method to Vicuna. All models compatible with FastChat can be implemented by just downloading them and madding the path to the weights in the script_extraction.py file.

