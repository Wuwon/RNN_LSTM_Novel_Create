from django.shortcuts import render,redirect
import argparse
import json
import os
from .sample import sample
# Create your views here.



def index(request):

    
    return render(request,'index.html')



def run(request):
    if request.method =='POST':
        content=request.POST['dahyun']
        sentense_generator = sample(100,header=content,num_chars=200)
        context ={
            'content' : content,
            'result' : sentense_generator
        }


        return render(request,'index.html',context)

    else:
        return render(request,'index.html')



