### CRAFT-MORAN_V2 Testing Pipeline

To run this please download the approriate models and put them in the models folder.

Download the neccessary packages (preferably in a conda virtual environment).

```
pip install -r requirements.txt
```

Then run the following command.

```
python test.py --trained_model=./models/craft_mlt_25k.pth --test_folder=./images
```

youll see the results in the results directory.