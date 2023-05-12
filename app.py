from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle

app=Flask(__name__)
car=pd.read_csv("cars24_clean.csv")
model= pickle.load(open("Cars24 linear model.pkl","rb"))



@app.route("/",methods=["GET","POST"])
def home():
    cars_brand=sorted(car["cars_brand"].unique())
    cars_name=sorted(car["cars_name"].unique())
    year=sorted(car["model_year"].unique())
    fuel_type=car["gasoliene_type"].unique()
    return render_template("index.html",cars_brand=cars_brand,cars_name=cars_name,year=year,fuel_type=fuel_type)


@app.route("/predict",methods=["GET"])
def predict_price():
    brand=request.args.get("company")
    carname=request.args.get("model")
    year=int(request.args.get("year"))
    fuel=request.args.get("fuel_type")
    dist=int(request.args.get("kms"))
    
    dic={"cars_name":carname,"cars_brand":brand,"model_year":year,"kms":dist,"gasoliene_type":fuel}
    result=model.predict(pd.DataFrame([dic]))
    return render_template("index.html",result =(np.round(result[0][0],2)))


if __name__=="__main__":
    app.run(debug=True)
