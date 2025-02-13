temp_threshold = 25
import cv2
import dht11
import RPi.GPIO as GPIO
import time
import telepot

receiverchatid = 1982112010
DHT_PIN = 4
sensor = dht11.DHT11(pin=DHT_PIN)
import os
import io

GPIO.setmode(GPIO.BCM)
r1 = 17
r2 = 27
ldr = 22
GPIO.setup(ldr, GPIO.IN)
GPIO.setup(r1, GPIO.OUT)
GPIO.output(r1, GPIO.HIGH)
GPIO.setup(r2, GPIO.OUT)
GPIO.output(r2, GPIO.HIGH)

frame_count = 0
thres = 0.45
temp = 0
hum = 0

def handle(msg):
    global telegramText, chat_id, receiveTelegramMessage
    chat_id = msg['chat']['id']
    telegramText = msg['text']
    if telegramText == "/start":
        msg = "your chat id " + str(chat_id)
        bot.sendMessage(chat_id, msg)
    else:
        receiveTelegramMessage = True

bot = telepot.Bot('7553105898:AAFvVvRmBoYAVqEenmNkuKLoMJa-BHKkQGs')
bot.message_loop(handle)
receiveTelegramMessage = False
sendTelegramMessage = False
statusText = ""

classNames = []
with open('coco.names', 'r') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

setfan = False
setlight = True
ldr_value = 0

def main():
    global cap
    message_sent = False
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(10, 70)
    cap.set(cv2.CAP_PROP_EXPOSURE, -12)
    start_time = time.time()
    
    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        ldr_value = GPIO.input(ldr)
        result = sensor.read()
        
        if result.is_valid():
            temp = result.temperature
            hum = result.humidity
        
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if classId == 1:
                    x, y, w, h = box
                    cv2.rectangle(img, box, color=(0, 0, 255), thickness=8)
                    message_sent = False
        
        if temp > temp_threshold:
            GPIO.output(r1, GPIO.LOW)
        else:
            GPIO.output(r1, GPIO.HIGH)
        
        if ldr_value == 0:
            GPIO.output(r2, GPIO.LOW)
        else:
            GPIO.output(r2, GPIO.HIGH)
        
        if message_sent == False:
            message_sent = True
            bot.sendMessage(receiverchatid, "Fan(200 w) and lights(100 w) turned off and you have saved 0.3 kw per hour")
        
        cv2.imshow("Output", img)
        print(time.time() - start_time)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
