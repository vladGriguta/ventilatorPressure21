# ventilatorPressure21


### Get the dataset
kaggle competitions download -c ventilator-pressure-prediction && unzip ventilator-pressure-prediction.zip -d data/ && rm ventilator-pressure-prediction.zip

### python setup
virtualenv venv -p /usr/bin/python3.8
source activate venv
pip install --upgrade pip && pip install -r requirements.txt

### submit solutions
kaggle competitions submit -c ventilator-pressure-prediction -f submission.csv -m "Message"