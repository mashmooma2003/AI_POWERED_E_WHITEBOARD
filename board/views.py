from django.shortcuts import render
import subprocess
import sys
import os

def home(request):
    return render(request, 'board/home.html')

def start_whiteboard(request):
    script_path = os.path.join(os.path.dirname(__file__), 'whiteboard.py')

    subprocess.Popen([sys.executable, script_path])

    return render(request, 'board/started.html')