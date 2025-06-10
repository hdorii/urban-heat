from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import requests
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import os
import json
import pandas as pd
from datetime import datetime
from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus
import urllib
load_dotenv()



app = Flask(__name__, template_folder="templates")
app.secret_key = os.urandom(24)
CORS(app)

def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    conn.autocommit = True
    return conn




API_KEY = unquote(os.getenv("SERVICE_KEY")) 

def get_temp_only(dt: datetime):
    url = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"

    # 전일까지만 제공 가능
    today = datetime.now()
    if dt.date() >= today.date():
        print("ASOS는 전일까지만 제공 가능")
        return None

    params = {
        "serviceKey": API_KEY,           
        "pageNo": "1",
        "numOfRows": "10",
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "HR",
        "startDt": dt.strftime("%Y%m%d"),
        "startHh": dt.strftime("%H"),
        "endDt": dt.strftime("%Y%m%d"),
        "endHh": dt.strftime("%H"),
        "stnIds": "108"  # 관악산
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
        if items:
            return float(items[0].get("ta"))

    except Exception as e:
        print("기상청 온도 조회 실패:", e)

    return None


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/get_temperature", methods=["POST"])
def get_temperature():
    data = request.get_json()
    timestamp_str = data.get("timestamp")

    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

        temperature = get_temp_only(dt)
        if temperature is not None:
            return jsonify({"temperature": round(temperature, 2)})
        else:
            return jsonify({"error": "온도 데이터를 찾을 수 없습니다."})
    except Exception as e:
        return jsonify({"error": f"예외 발생: {str(e)}"})


@app.route("/predict_result", methods=["POST"])
def predict_result():
    input_data = request.get_json()

    try:
        dt_obj = datetime.fromisoformat(input_data["timestamp"].replace("Z", "+00:00"))
    except Exception as e:
        return jsonify({"error": f"timestamp 변환 오류: {str(e)}"}), 400

    temp = get_temp_only(dt_obj)
    if temp is None:
        return jsonify({"error": "해당 시간의 기온 데이터를 가져올 수 없습니다."}), 500
    input_data["suburban_temp_current"] = temp

    input_data["timestamp"] = int(dt_obj.timestamp())

    columns = [
        "District",
        "green_rate",
        "Building_Density",
        "car_registration_count",
        "population_density",
        "avg_km_per_road_km",
        "timestamp",
        "suburban_temp_current"
    ]

    try:
        data_row = [input_data[col] for col in columns]
    except KeyError as e:
        return jsonify({"error": f"입력값 누락: {str(e)}"}), 400

    payload = {
        "dataframe_split": {
            "columns": columns,
            "data": [data_row]
        }
    }

    headers = {
        "Authorization": DATABRICKS_TOKEN,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(DATABRICKS_MODEL_URL, headers=headers, json=payload)
        prediction_result = response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

   
@app.route("/api/get_temp_by_timestamp")
def api_get_temp():
    timestamp = request.args.get("timestamp")
    try:
        dt = datetime.fromisoformat(timestamp)
    except Exception:
        return jsonify({"error": "올바른 timestamp 형식이 아닙니다."}), 400

    temp = get_temp_only(dt)
    if temp is None:
        return jsonify({"error": "해당 시간의 기온을 가져올 수 없습니다."}), 500
    return jsonify({"temperature": temp})


@app.route("/features")
def features():
    return render_template("features.html")

@app.route("/heatmap_view")
def heatmap_view():
    return render_template("heatmap.html")

@app.route("/api/available_times")
def available_times():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT year, month, day, hour
        FROM uhii.part
        ORDER BY year, month, day, hour
    """)
    times = cur.fetchall()
    result = [f"{y:04d}-{m:02d}-{d:02d} {h:02d}:00" for (y, m, d, h) in times]
    return jsonify(result)


@app.route('/power_bi')
def show_report():
    return render_template("power_bi.html")





@app.route("/api/heatmap_by_time")
def heatmap_by_time():
    timestamp = request.args.get("timestamp")
    if not timestamp:
        return jsonify({"error": "timestamp required"}), 400

    try:
        y, m, d = map(int, timestamp[:10].split("-"))
        h = int(timestamp[11:13])
    except:
        return jsonify({"error": "invalid timestamp format"}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT "District", "UHII"
        FROM uhii.part
        WHERE year = %s AND month = %s AND day = %s AND hour = %s
    """, (y, m, d, h))
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["district", "uhii"])

    # 영문 district → 한글 district 매핑
    district_name_map = {
        "Gangnam-gu": "강남구",
        "Gangdong-gu": "강동구",
        "Gangbuk-gu": "강북구",
        "Gangseo-gu": "강서구",
        "Gwanak-gu": "관악구",
        "Gwangjin-gu": "광진구",
        "Guro-gu": "구로구",
        "Geumcheon-gu": "금천구",
        "Nowon-gu": "노원구",
        "Dobong-gu": "도봉구",
        "Dongdaemun-gu": "동대문구",
        "Dongjak-gu": "동작구",
        "Mapo-gu": "마포구",
        "Seodaemun-gu": "서대문구",
        "Seocho-gu": "서초구",
        "Seongdong-gu": "성동구",
        "Seongbuk-gu": "성북구",
        "Songpa-gu": "송파구",
        "Yangcheon-gu": "양천구",
        "Yeongdeungpo-gu": "영등포구",
        "Yongsan-gu": "용산구",
        "Eunpyeong-gu": "은평구",
        "Jongno-gu": "종로구",
        "Jung-gu": "중구",
        "Jungnang-gu": "중랑구"
    }

    # 영문 이름을 한글로 매핑
    df["district_kor"] = df["district"].map(district_name_map)

    with open("data/seoul_gu_25.geojson", encoding="utf-8") as f:
        geojson = json.load(f)



    for feature in geojson["features"]:
        name = feature["properties"]["sggnm"]  # 한글 구 이름
        match = df[df["district_kor"] == name]

        if not match.empty and pd.notnull(match["uhii"].values[0]):
            feature["properties"]["uhii"] = float(match["uhii"].values[0])
        else:
            feature["properties"]["uhii"] = None

        # 프론트엔드에서 쉽게 사용하도록 구 이름을 명시적으로 추가
        feature["properties"]["name"] = name


    return jsonify(geojson)

if __name__ == "__main__":
    app.run(debug=True)
