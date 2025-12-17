# Python 


**Python Manager**
    https://www.python.org/downloads/release/pymanager-252/

* 다양한 Python Version 사용목적 
```
python --version
py --version
```

```
where python
where py
```


## Python venv 


* setup venv

```
python -m venv .venv
or 
py -3.11 -m venv .venv311
```


```
pip install -r .\requirements.txt
```


## Window venv Activate


* PowerShell (관리자)   
```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

* Activate 
```
.venv\Scripts\Activate.ps1
or
.venv311\Scripts\Activate.ps1
```

python cxl_phy_viz_2.py

## Error 


Not used PyQt6~~~

* Error
```
file:///D:/works/projects/phy_simulation/simulation/cxl_phy_viz_2.qml:2:1: 라이브러리 D:\works\projects\phy_simulation\simulation\.venv\Lib\site-packages\PySide6\qml\QtQuick\qtquick2plugin.dll을(를) 불러올 수 없음: 지정된 모듈을 찾을 수 없습니다.
```

* VC_redist.x64
    https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170 







