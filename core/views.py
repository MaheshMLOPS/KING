from django.contrib import messages
from django.http.response import JsonResponse
from django.shortcuts import get_object_or_404, render
from record import settings
from .models import Record
import os





BASE_DIR=str(settings.BASE_DIR)

import pandas as pd
from nltk.tokenize import word_tokenize as wt
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from rake_nltk import Rake
import numpy as np
import whisper
import moviepy.editor as mp
from joblib import load


df=pd.read_csv(BASE_DIR+r"/videos/SpacyCSV.csv")
print(df)
r=Rake()
sentences=df['Questions']
keywords=[]
for sentence in sentences:
    r.extract_keywords_from_text(sentence)
    keywords.append(r.get_ranked_phrases())

def getkeywords(sentences):
    keywords=[]
    r.extract_keywords_from_text(sentences)
    keywords.append(r.get_ranked_phrases())
    return keywords


model = KeyedVectors.load_word2vec_format(datapath(BASE_DIR+r"/videos/GoogleNews-vectors-negative300.bin"), binary=True)
path=str(BASE_DIR)
clf = load(BASE_DIR+r"/videos/decisionclf.pickle") 

vector=[]
def listit(v):
    x=[]
    for i in v:
        for j in i:
            x.append(j)
    return x
def getvec(keywords):
    vectors=[]
    def listit(v):
        x=[]
        for i in v:
            for j in i:
                x.append(j)
        return x
    for i in keywords:
        i=listit(i)
        vector=[]
        for j in i:
            try:
                vector.append(model[j])
            except:
                continue
        vectors.append(vector)
    return vectors
vectors=getvec(keywords)
def fill(v):
    value=np.zeros(shape=(100,300))
    value[0:len(v)]=v
    return value






def record(request):
    if request.method == "POST":
        audio_file = request.FILES.get("recorded_audio")
        language = request.POST.get("language")
        record = Record.objects.create(language=language, voice_record=audio_file)
        record.save()
        messages.success(request, "Audio recording successfully added!")
        return JsonResponse(
            {
                "url": record.get_absolute_url(),
                "success": True,
            }
        )
    context = {"page_title": "Record audio"}
    return render(request, "core/record.html", context)


def record_detail(request, id):
    records = Record.objects.all()
    link=" "
    d={1:'What_fascinating_about_space',2:'So_actually_I_wanted_to_know_that_what_is_the_what_is_exactly_the_work_that_you_all',3:"I_want to go into research and astrophysics and I'm in the 10th grade, path should I follow to get there What is the main career like",4:"I wanted to ask what do you think that space space found this was And what way expectations is as a yes, education stranger that the same",5:"chemistry is also necessary along with maths and physics in 11 terms Is that true",6:"Is it what role does chemistry play in this  space field"}
    model = whisper.load_model("base")
    print('hi')
    filename=os.listdir(path+r'\media\records')[0] 
    audio_file=path+r'\media\records'+'\\'+filename
    name=path+r'\media\records'+'\\'+filename.split(".")[0]
    dest=''
    dest+=str(name)+str('.mp3')
    print('ource is = ',audio_file,' destinatoin is = ',dest)
    os.rename(audio_file, dest)
    audio_file=dest
    print('model file is ',audio_file)
    lin=model.transcribe(audio_file)['text']
    print(lin)
    keywords=getkeywords(lin)
    predictit=getvec(keywords)
    predictit1=fill(predictit[0])
    link=d[clf.predict(predictit1.reshape(1,-1))[0]]
    os.remove(audio_file)
    context = {
        "page_title": "Recorded audio detail",
        "records": records,
        'video':str(path+"\\videos\\"+link+".mp4"),
    }
    records.delete()
    return render(request, "core/record_detail.html", context)
# def index(request):
#     records = Record.objects.all()
#     context = {"page_title": "Voice records", "records": records}
#     return render(request, "core/index.html", context)
# 
