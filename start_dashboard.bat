@echo off
cd /d "%~dp0docs"
start "" http://localhost:8000/index.html
echo Dashboard chal raha hai: http://localhost:8000  (band karne ke liye ye window band karo)
python -m http.server 8000
