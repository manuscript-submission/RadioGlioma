"""
Adapted from FastChat (LMSys) by Bastien Le Guellec for research purposes
"""

import time
import csv
from itertools import zip_longest

timestart=time.time()

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

import os
import sys

from fastchat.model.model_adapter import add_model_args
from fastchat.serve.inference import ChatIO, chat_loop_finding, chat_loop_oxford, chat_loop_berden, chat_loop_gliome, chat_loop_lupus, chat_loop_lupus_class, chat_loop_gliome_indic

vicuna = '***/vicuna-13b/'
vicuna1_3 = '***/lmsys_vicuna-13b-v1.3/'
vicuna1_5 = '***/lmsys_vicuna-13b-v1.5-16k/'

vicuna33_1_3='***/lmsys_vicuna-33b-v1.3/'

medalpaca = '***/medalpaca_medalpaca-13b/'
koala = '***/vicuna-13b/'


f_gliome='/***.txt'
f_gliome_indic='/***.txt'


shots_gliome=[
['Doctor', '''I will present you a radiology report from a patient with a glioma. I want you to extract the size from the target lesions in the report and in the previous exam. It is crucial that you report only taget ("cible") lesions. There may be no target lesion. You must respond according to this template: 
- Target X today: [size in the report]
- Target X previously: [previous size]
(Create a new line for every new measurement. Include ONLY lesions that are explicit TARGET)
 
Here are a few examples :

Doctor: 
Lésion cible : Discrète majoration de la prise de contraste nodulaire en regard de l’infiltration lésionnelle flair située au contact de la corne frontale du ventricule latéral gauche mesurée à 22 x 10 mm dans le plan coronal contre 20 x 10 mm précédemment. Absence d'hyperperfusion en regard de cette prise de contraste. Meilleure visibilité d’une prise de contraste nodulaire apparue sur l’IRM précédente au sein de la substance blanche frontale gauche à proximité de l’infiltration lésionnelle décrite ci-dessus, mesurée à 7 mm de grand axe contre 5 mm précédemment. Absence d’hyperperfusion au sein de cette prise de contraste. Stabilité de l'infiltration lésionnelle en hypersignal flair fronto-cingulaire gauche. Cavité opératoire fronto-cingulaire droite ouverte au sein de la corne frontale avec stabilité de la prise de contraste linéaire de sa berge postérieure et des hypersignaux flair au pourtour. Multiples microsaignements anciens de la substance blanche sous corticale du vertex en rapport avec des cavernomes post radiques. 
Robot:
- Target 1 today: 22x10
- Target 1 previously: 20x10
 
Doctor: 
Résultats comparaison à l'IRM du 10 février 2022. IRM 10/02/22 IRM19/05/22 Lésions cibles et non cibles : Nette majoration de l'infiltration lésionnelle cortico-sous-corticale en hypersignal flair avec prise de contraste et hyperperfusion (rCBV max: 3,8) en son sein intéressant la région frontale droite, s'étendant au corps calleux (genou et tiers antérieur), au striatum et à la région insulaire antérieure : - Lésion fronto-calleuse droite (cible 1) : 56 x 41 mm contre 36 x 45 mm précédemment ; - Lésion striatum droite (cible 2) : 36 x 33 mm contre 12 x 16 mm ; - Lésion frontale supérieure droite (cible 3) : 30 x 35 mm (non mesurable précédemment, lésion non cible). Apparition d'un effet de masse sur les cornes frontales des ventricules latéraux. Stabilité de la déviation millimétrique de la ligne médiane vers la droite. Remaniements post thérapeutique : Stabilité des plages de leucopathie de la substance blanche profonde et périventriculaire fronto-pariétale bilatérale. Stabilité de la fine prise de contraste pachyméningée frontotemporale droite en regard de la voie d'abord. 
Robot:
- Target 1 today: 56x41
- Target 1 previously: 36x45
- Target 2 today: 36x33
- Target 2 previsouly: 12x16
- Target 3 today: 30x35
- Target 3 previously: non mesurable
 
Doctor: 
Résultats comparaison à l'IRM du 16 mai 2022. Lésion cible : -Majoration en taille de la lésion temporale supérieure gauche mesurée à 14 x 11 mm dans le plan coronal contre 11 mm de grand axe précédemment ainsi que de la plage d'œdème aux pourtours. Lésions non cible : Stabilité en taille de la lésion non cible à centre nécrotique temporo-insulaire gauche mesurée à 29 x 30 mm dans le plan axial avec diminution de l'épaisseur des prises de contraste pariétales et de la plage d'œdème aux pourtours. Stabilité en taille du système ventriculaire. Lésion pariétale supérieure gauche rehaussée : mesurant 8,9 x 6,8 mm et présentant une hyperperfusion avec rCBV corr de 2,7. Éléments médians en place. Aspect inchangé des discrets hypersignaux flair focaux non spécifiques de la substance périventriculaire et sous corticale bifrontale. Stabilité en taille de la lésion rehaussée extra-axiale frontale moyenne droite de 4 mm de grand axe compatible avec un méningiome. 
Robot:
- Target 1 today: 14x11
- Target 1 previously: 11
 
Doctor:
Résultats comparaison aux IRM antérieures de 2022. Lésions cibles : absence Lésions non cible : Stabilité de l'infiltration lésionnelle en hypersignal flair intéressant les régions insulaire antérieure, frontale interne et fronto-orbitaire postérieure gauches ainsi que le genou du corps calleux. Meilleure visibilité d'une prise de contraste linéaire apparue sur l'IRM du 21 novembre 2022, non spécifique, située en profondeur de l'infiltration lésionnelle insulaire, au contact d'une formation kystique (cf. Images clés sur le PACS). Remaniements post thérapeutiques : Stabilité de la plage en hypersignal flair frontale supérieure gauche, sans hyperperfusion ni restriction de diffusion en son sein et n’atteignant pas le cortex, compatible avec un remaniement post-thérapeutique. Stabilité des fine prises de contraste des parois de la cavité opératoire frontale gauche d'allure réactionnelle. Stabilité en taille du système ventriculaire. 
Robot:
- No target lesion
'''],
['Chatbot', "Understood, I will respond according to the template."]
]

shots_gliome_indic=[
['Doctor', '''I will present you a radiology report from a patient with a glioma. I want you to extract information from the indication section. You must respond according to this template: 
- Surgery ?: [yes or no (biopsy is not surgery)]
- Radiation therapy ?: [yes or no]
- Chemotherapy ?: [yes or no ?]
- Mutation ?: [none mentioned/negative ones/positive ones (when MGMT is hypermethylated it is positive)]
 
Here are a few examples :

Doctor: 
Indication:
Surveillance d'un glioblastome diagnostiqué sur des biopsies stéréotaxiques puis pris en
charge par radiochimiothérapie concomitante de juin à juillet 2020 suivie de 6 cycles de
Témodal jusqu'en janvier 2021 associée à de l'Avastin d'octobre 2020 à mars 2021.
Progression sur l'IRM de juin 2021 avec reprise de l’Avastin jusqu'en juillet 2021
(traitement arrêté en raison d'une cytolyse hépatique).
Robot:
- Surgery ?: no
- Radiation therapy ?: yes
- Chemotherapy ?: yes
- Mutation ?: none mentioned
 
Doctor: 
Indication :
Surveillance chez une patiente aux antécédents de glioblastome frontal supérieur gauche non hyperméthylé MGMT, avec amplification du gène EGFR. Prise en charge initiale par
chirurgie en mars 2018 puis radiochimiothérapie concomitante et 6 cycles de Témodal jusqu'en mars 2019.
Progression en janvier 2022 prise en charge par Témodal.
Robot:
- Surgery ?: yes
- Radiation therapy ?: yes
- Chemotherapy ?: yes
- Mutation ?: MGMT negative / EGFR positive
 
Doctor:
Indication:
Surveillance dans le cadre d'un astrocytome IDH1 non muté de grade IV diagnostiqué sur des biopsies.
Radiochimiothérapie selon le protocole STUPP avec fin de la radiothérapie le 21 janvier 2022.
Chimiothérapie adjuvante par Témodal.
Robot:
- Surgery ?: no
- Radiation therapy ?: yes
- Chemotherapy ?: yes
- Mutation ?: IDH1 negative
'''],
['Chatbot', "Understood, I will respond according to the template."]
]



class SimpleChatIO(ChatIO):
    def __init__(self, multiline: bool = False):
        self._multiline = multiline

    def prompt_for_input(self, role) -> str:
        if not self._multiline:
            return input(f"{role}: ")

        prompt_data = []
        line = input(f"{role} [ctrl-d/z on empty line to end]: ")
        while True:
            prompt_data.append(line.strip())
            try:
                line = input()
            except EOFError as e:
                break
        return "\n".join(prompt_data)

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)

    def print_output(self, text: str):
        print(text)

chatio=SimpleChatIO



d=chat_loop_gliome(model_path=vicuna1_5,device='cuda',num_gpus=3,max_gpu_memory='30Gib',load_8bit=False,cpu_offloading=False,conv_template="vicuna_v1.1",conv_system_msg=None,temperature=0,repetition_penalty=100,max_new_tokens=1024,chatio=chatio,debug=False,few_shots=shots_gliome,file_path=f_gliome, gptq_config=False, revision=False)

export_data = zip_longest(*d, fillvalue = '')

with open('/***.csv', 'w', newline='') as myfile:
      wr = csv.writer(myfile)

      wr.writerows(export_data)
myfile.close()
