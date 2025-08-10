import math
import random
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# 🎥 Initialize Camera
camera = cv2.VideoCapture(0)
camera.set(3, 1280)  # Width
camera.set(4, 720)   # Height

# ✋ Hand detector instance
handTracker = HandDetector(detectionCon=0.8, maxHands=1)


class GestureSnake:
    def __init__(self, foodImagePath):
        # Snake body coordinates
        self.bodyCoords = []
        self.segmentDistances = []
        self.totalLength = 0
        self.maxLength = 150
        self.lastHeadPos = (0, 0)

        # Load food image
        self.foodSprite = cv2.imread(foodImagePath, cv2.IMREAD_UNCHANGED)
        self.foodHeight, self.foodWidth, _ = self.foodSprite.shape
        self.foodLocation = (0, 0)
        self._placeFood()

        # Score & game status
        self.score = 0
        self.isGameOver = False

    def _placeFood(self):
        """Randomly place food on the screen."""
        self.foodLocation = (
            random.randint(100, 1000),
            random.randint(100, 600)
        )

    def updateFrame(self, frame, headPosition):
        """Update game state and render."""
        if self.isGameOver:
            cvzone.putTextRect(frame, "Game Over", [300, 400],
                               scale=7, thickness=5, offset=20)
            cvzone.putTextRect(frame, f'Final Score: {self.score}', [300, 550],
                               scale=7, thickness=5, offset=20)
        else:
            prevX, prevY = self.lastHeadPos
            headX, headY = headPosition

            # Add new head position
            self.bodyCoords.append([headX, headY])
            distance = math.hypot(headX - prevX, headY - prevY)
            self.segmentDistances.append(distance)
            self.totalLength += distance
            self.lastHeadPos = (headX, headY)

            # Shrink snake if it exceeds max length
            while self.totalLength > self.maxLength and self.segmentDistances:
                self.totalLength -= self.segmentDistances.pop(0)
                self.bodyCoords.pop(0)

            # Check food collision
            foodX, foodY = self.foodLocation
            if (foodX - self.foodWidth // 2 < headX < foodX + self.foodWidth // 2 and
                    foodY - self.foodHeight // 2 < headY < foodY + self.foodHeight // 2):
                self._placeFood()
                self.maxLength += 50
                self.score += 1
                print(f"Score: {self.score}")

            # Draw snake
            for idx in range(1, len(self.bodyCoords)):
                cv2.line(frame, tuple(self.bodyCoords[idx - 1]),
                         tuple(self.bodyCoords[idx]), (0, 0, 255), 20)
            if self.bodyCoords:
                cv2.circle(frame, tuple(self.bodyCoords[-1]),
                           20, (0, 255, 0), cv2.FILLED)

            # Draw food
            frame = cvzone.overlayPNG(frame, self.foodSprite,
                                      (foodX - self.foodWidth // 2, foodY - self.foodHeight // 2))

            # Display score
            cvzone.putTextRect(frame, f'Score: {self.score}', [50, 80],
                               scale=3, thickness=3, offset=10)

            # Collision with self
            if len(self.bodyCoords) > 4:
                pts = np.array(self.bodyCoords[:-2], np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 0), 3)
                minDist = cv2.pointPolygonTest(pts, (headX, headY), True)
                if -1 <= minDist <= 1:
                    print("Self collision detected!")
                    self._resetGame()

        return frame

    def _resetGame(self):
        """Reset game state after collision."""
        self.isGameOver = True
        self.bodyCoords.clear()
        self.segmentDistances.clear()
        self.totalLength = 0
        self.maxLength = 150
        self.lastHeadPos = (0, 0)
        self._placeFood()


# 🐍 Game object
snakeGame = GestureSnake(r"D:\snakegame\.venv\img.png")

while True:
    ret, frame = camera.read()
    if not ret:
        print("⚠️ Webcam not detected. Check camera index.")
        continue

    frame = cv2.flip(frame, 1)  # Mirror view
    detectedHands, frame = handTracker.findHands(frame, flipType=False)

    if detectedHands:
        fingerTip = detectedHands[0]['lmList'][8][0:2]  # Index finger tip coords
        frame = snakeGame.updateFrame(frame, fingerTip)

    cv2.imshow("Gesture Snake", frame)
    keyPress = cv2.waitKey(1)
    if keyPress == ord('r'):
        snakeGame.isGameOver = False
