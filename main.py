import os
import io
import random
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from typing import Dict
from google.cloud import vision
from fastapi.staticfiles import StaticFiles

# Render의 비밀 변수(Environment Variable)를 읽어와 파일처럼 사용하도록 설정
gcp_credentials_content = os.environ.get('GCP_CREDENTIALS_JSON')
if gcp_credentials_content:
    # Render 서버에서만 임시 파일을 생성합니다.
    with open('gcp-credentials-temp.json', 'w', encoding='utf-8') as f:
        f.write(gcp_credentials_content)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp-credentials-temp.json'
else:
    # 로컬 PC에서 테스트할 때는 기존 파일을 사용합니다.
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp-credentials.json'


vision_client = vision.ImageAnnotatorClient()
app = FastAPI()
app.mount("/static", StaticFiles(directory="images"), name="static")


# --- 캐릭터 데이터베이스 ---
CHARACTERS_DATA = {
    "denji": {"name": "덴지", "archetype": "chaotic", "reason": "밝고 즐거운 에너지가 넘쳐요!", "image_url": "/static/denji.jpg"},
    "power": {"name": "파워", "archetype": "chaotic", "reason": "강렬하고 예측할 수 없는 매력이 있어요!", "image_url": "/static/power.jpg"},
    "beam": {"name": "빔", "archetype": "chaotic", "reason": "맹목적일 정도의 순수한 열정이 느껴져요!", "image_url": "/static/beam.jpg"},
    "pochita": {"name": "포치타", "archetype": "chaotic", "reason": "세상에서 가장 귀여운 혼돈을 보여줘요!", "image_url": "/static/pochita.jpg"},
    "aki": {"name": "아키", "archetype": "professional", "reason": "차분하고 깊은 감성이 느껴져요.", "image_url": "/static/aki.jpg"},
    "kishibe": {"name": "키시베", "archetype": "professional", "reason": "모든 걸 꿰뚫어 보는 듯한 관록이 있어요.", "image_url": "/static/kishibe.jpg"},
    "quanxi": {"name": "콴시", "archetype": "professional", "reason": "나른함 속에 숨겨진 최강의 실력자 같아요.", "image_url": "/static/quanxi.jpg"},
    "makima": {"name": "마키마", "archetype": "mysterious", "reason": "속을 알 수 없는 신비로운 분위기가 느껴져요.", "image_url": "/static/makima.jpg"},
    "yoshida": {"name": "요시다", "archetype": "mysterious", "reason": "미소 뒤에 무언가 숨기고 있는 것 같아요.", "image_url": "/static/yoshida.jpg"},
    "kobeni": {"name": "코베니", "archetype": "anxious", "reason": "어딘가 모르게 지켜주고 싶은 불안함이 보여요.", "image_url": "/static/kobeni.jpg"},
    "himeno": {"name": "히메노", "archetype": "hedonist", "reason": "체념한 듯한 어른의 매력이 느껴져요.", "image_url": "/static/himeno.jpg"},
    "angel": {"name": "엔젤", "archetype": "hedonist", "reason": "모든 게 귀찮은 듯한 나른함이 매력적이에요.", "image_url": "/static/angel.jpg"},
    "reze": {"name": "레제", "archetype": "duality", "reason": "순수한 모습 뒤에 다른 얼굴이 숨어있을 것 같아요.", "image_url": "/static/reze.jpg"},
    "asa": {"name": "아사", "archetype": "duality", "reason": "평범한 일상 속, 특별한 비밀을 간직하고 있군요.", "image_url": "/static/asa.jpg"}
}


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/api/analyze-face")
async def analyze_face(file: UploadFile = File(...)) -> Dict[str, str]:
    content = await file.read()
    image = vision.Image(content=content)

    features = [
        vision.Feature(type_=vision.Feature.Type.FACE_DETECTION),
        vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION),
        vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
    ]
    request = vision.AnnotateImageRequest(image=image, features=features)
    response = vision_client.annotate_image(request=request)

    scores = {
        "chaotic": 0, "professional": 0, "mysterious": 0,
        "anxious": 0, "hedonist": 0, "duality": 0
    }

    if response.face_annotations:
        face = response.face_annotations[0]
        if face.joy_likelihood >= 4 or face.anger_likelihood >= 3: scores["chaotic"] += 15
        if face.sorrow_likelihood >= 3 or face.surprise_likelihood >= 3: scores["anxious"] += 15
        if face.joy_likelihood >= 2 and face.sorrow_likelihood >= 2: scores["duality"] += 15
        if face.headwear_likelihood > 2: scores["hedonist"] += 15
        
        emotion_sum = sum([face.joy_likelihood, face.sorrow_likelihood, face.anger_likelihood, face.surprise_likelihood])
        if emotion_sum <= 4:
            scores["professional"] += 10
            scores["mysterious"] += 5

    if response.label_annotations:
        for label in response.label_annotations:
            if label.description.lower() in ["glasses", "eyewear", "sunglasses"]:
                scores["professional"] += 10
            
            # --- [오류 수정된 부분!] ---
            if label.description in ["sky", "crowd", "outdoor"]: scores["chaotic"] += 5
            if label.description in ["room", "office", "building"]: scores["professional"] += 5
            if label.description in ["night", "darkness"]: scores["mysterious"] += 10
            if label.description in ["cafe", "restaurant", "food"]: scores["hedonist"] += 5
            if label.description in ["school", "street", "book"]: scores["duality"] += 5

    if response.image_properties_annotation and response.image_properties_annotation.dominant_colors:
        dominant_colors = response.image_properties_annotation.dominant_colors.colors
        warm_colors, cool_colors = 0, 0
        for color_info in dominant_colors:
            color = color_info.color
            if color.red > color.blue and color.red > color.green: warm_colors += color_info.pixel_fraction
            if color.blue > color.red: cool_colors += color_info.pixel_fraction
        if warm_colors > 0.5: scores["chaotic"] += 10
        if cool_colors > 0.5: scores["professional"] += 10

    if not any(scores.values()):
        winning_archetype = "mysterious"
    else:
        winning_archetype = max(scores, key=scores.get)

    candidate_chars = [char for char, data in CHARACTERS_DATA.items() if data["archetype"] == winning_archetype]
    
    if not candidate_chars:
        selected_char_name = "pochita" if winning_archetype == "chaotic" else "makima"
    else:
        selected_char_name = random.choice(candidate_chars)

    return CHARACTERS_DATA[selected_char_name]