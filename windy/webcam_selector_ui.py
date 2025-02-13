import streamlit as st
import requests
import json
import pandas as pd
from os import path

import requests

@st.cache()
def get_key():
    with open("token.txt","r") as f:
        key = f.readline().strip()
    return key

@st.cache()
def get_cams(offset=0):
    baseurl = "https://api.windy.com/api/webcams/v2/list/"
    key = get_key()
    keystring = f"?key={key}"
    show = "webcams:location,image,url"
    category = "city"
    #params = f"limit=25,{offset}/country=DE/category={category}/orderby=popularity,desc"
    params = f"limit=25,{offset}/country=DE"
    url = f"{baseurl}{params}?key={key}&show={show}"
    print(url)
    r=requests.get(url)
    print(r,r.json()["status"])

    returned = len(r.json()["result"]["webcams"])
    total = r.json()["result"]["total"]
    #print(f"Webcams: {returned}/{total}")

    webcams = r.json()["result"]["webcams"]
    return webcams,total

COUNTER = int(st.number_input("Select",min_value=0,step=1,value=0))
ADD = st.button("Add this camera")
offset = 25*(COUNTER//25)
idx = COUNTER%25
webcams,total = get_cams(offset)
st.write(f"Position: {COUNTER}/{total} (offset={offset}, idx={idx})")
webcam = webcams[idx]
url = webcam["image"]["daylight"]["preview"]
st.write(webcam["title"])
st.image(url)
JSONFILE = "output.json"
if path.exists(JSONFILE):
    df = pd.read_json(JSONFILE)
    df = df.drop_duplicates(subset="windy_id",keep="last")
else:
    # create empty file with utf-8 encoding
    with open(JSONFILE,"w+",encoding="utf-8") as f:
        f.write("[]")
    df = pd.DataFrame()
my_table = st.table(df)

if ADD:
    print(f"Add id={idx}: {webcam['title']}")
    data = {
        "name":webcam["title"],
        "lat":webcam["location"]["latitude"],
        "lon":webcam["location"]["longitude"],
        "windy_preview_url":webcam["image"]["current"]["preview"],
        "windy_url":webcam["url"]["current"]["desktop"],
        "windy_id":webcam["id"],
        "selector_counter":COUNTER
    }
    df_add = pd.DataFrame(data=data,index=[0])
    df = df.append(df_add)
    df["windy_id"]=df["windy_id"].astype(int)
    df = df.drop_duplicates(subset="windy_id",keep="last").reset_index(drop=True)
    
    json_str=df.to_json(orient="records",indent=2,force_ascii=False)
    json_str=json_str.replace("\/","/") # un-escpae slashes
    with open(JSONFILE,"w",encoding="utf-8") as f:
        f.write(json_str)
        # for some reason, pd.to_json(JSONFILE) creates issues with utf8 files
    
    st.write("Source added!")
    my_table.table(df)
    

st.markdown("""
<style type='text/css'>
    .stButton>button {
        width:100% !important;
        height:5em;
        background:#cfc !important;
    }
    .stButton>button:hover {
        background:#dfd !important;
    }
</style>
""", unsafe_allow_html=True)
