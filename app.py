
import cv2
import os
import time
import winsound
from datetime import datetime
import subprocess
from flask import Flask, render_template, Response, request, redirect, session
last_alert_time = 0
ALERT_COOLDOWN = 30

LOG_FILE = "logs/access_log.txt"
os.makedirs("logs", exist_ok=True)

def log_event(status):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {status}\n"
        )

app = Flask(__name__)
app.secret_key = "securesight_ai_2026"
os.makedirs("static/evidence/alert", exist_ok=True)
os.makedirs("static/evidence/suspicious", exist_ok=True)

# ======================
# LOAD TRAINED MODEL
# ======================

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

label_map = {
    0: "bharat",
    1: "criminal1"
    
}

# ======================
# FACE DETECTOR
# ======================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

# ======================
# CAMERA STREAM
# ======================

def generate_frames():

    global last_alert_time

    while True:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))

            label, confidence = recognizer.predict(face_roi)

            current_time = datetime.now().strftime("%H:%M:%S")

            if confidence < 80:

                name = label_map.get(label, "Unknown")

                if name == "bharat":

                    text = "AUTHORIZED : Bharat"
                    color = (0, 255, 0)

                elif name == "criminal1":

                    text = "CRIMINAL : Criminal1"
                    color = (0, 0, 255)

                    current = time.time()

                    if current - last_alert_time > ALERT_COOLDOWN:

                        filename = f"static/evidence/alert/{int(current)}.jpg"
                        saved = cv2.imwrite(filename, frame)
                        print("CRIMINAL SAVED:", saved, filename)

                        log_event("CRIMINAL DETECTED")

                        try:
                            winsound.PlaySound(
                                "siren.wav",
                                winsound.SND_ASYNC
                            )
                        except:
                            pass

                        last_alert_time = current

                else:

                    text = "UNKNOWN"
                    color = (0, 165, 255)

            else:

                text = "UNKNOWN PERSON"
                color = (0, 165, 255)

                filename = f"static/evidence/suspicious/{int(time.time())}.jpg"

                saved = cv2.imwrite(filename, frame)
                print("UNKNOWN SAVED:", saved, filename)

                log_event("UNKNOWN PERSON")

            cv2.putText(
                frame,
                current_time,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

# ======================
# ROUTES
# ======================

@app.route("/")
def dashboard():

    if not session.get("logged_in"):
        return redirect("/login")

    return render_template("dashboard.html")

@app.route("/stats")
def stats():

    alert_count = len(os.listdir("static/evidence/alert"))
    suspicious_count = len(os.listdir("static/evidence/suspicious"))

    return {
        "criminal_alerts": alert_count,
        "unknown_persons": suspicious_count
    }

@app.route("/retrain")
def retrain():

    subprocess.run(
        ["python", "train_model.py"]
    )

    return "Model Retrained Successfully"

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "admin123":

            session["logged_in"] = True

            return redirect("/")

    return render_template("login.html")

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")

@app.route("/add_criminal", methods=["GET", "POST"])
def add_criminal():

    if request.method == "POST":

        name = request.form["name"]

        folder = os.path.join(
            "danger_dataset",
            name
        )

        os.makedirs(folder, exist_ok=True)

        files = request.files.getlist("images")

        count = 1

        for file in files:

            if file.filename:

                path = os.path.join(
                    folder,
                    f"{count}.jpg"
                )

                file.save(path)

                count += 1

        return redirect("/criminal_database")

    return render_template(
        "add_criminal.html"
    )

@app.route("/logs")
def logs():

    data = []

    if os.path.exists(LOG_FILE):

        with open(LOG_FILE, "r", encoding="utf-8") as f:

            lines = f.readlines()

            data = lines[-20:]

    return {
        "logs": data
    }
@app.route("/evidence")
def evidence():

    alert_images = os.listdir("static/evidence/alert")
    suspicious_images = os.listdir("static/evidence/suspicious")

    return render_template(
        "evidence.html",
        alert_images=alert_images,
        suspicious_images=suspicious_images
    )


@app.route("/criminal_database")
def criminal_database():

    criminals = os.listdir("danger_dataset")

    return render_template(
        "criminal_database.html",
        criminals=criminals
    )

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ======================
# RUN
# ======================

if __name__ == "__main__":
    app.run(debug=True)