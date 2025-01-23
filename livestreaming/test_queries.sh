#!/bin/bash

LIVESTREAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LECTURE_DIR="$LIVESTREAM_DIR/lectures"

LECTURE_NUMBER=$1

if [ -z "$LECTURE_NUMBER" ]; then
  echo "Usage: bash test_queries.sh <lecture_number>"
  exit 1
fi

CONDA_ENV="askademia"
VIDEO_PREFIX="video"
DURATION=5400

VIDEO_FILE="$LECTURE_DIR/${VIDEO_PREFIX}${LECTURE_NUMBER}.mp4"
if [ ! -f "$VIDEO_FILE" ]; then
  echo "Error: Video file '$VIDEO_FILE' not found."
  exit 1
fi

# Start processes in separate Terminal windows
osascript -e "tell application \"Terminal\" to do script \"node-media-server\"" &
NODE_MEDIA_SERVER_PID=$!

sleep 3

osascript -e "tell application \"Terminal\" to do script \"cd $LECTURE_DIR && conda activate $CONDA_ENV && ffmpeg -re -i ${VIDEO_PREFIX}${LECTURE_NUMBER}.mp4 -c copy -f flv rtmp://localhost/live/my_stream\"" &
FFMPEG_PID=$!

sleep 1

osascript -e "tell application \"Terminal\" to do script \"cd $LIVESTREAM_DIR && conda activate $CONDA_ENV && python app.py\"" &
APP2_PID=$!

sleep 5

osascript -e "tell application \"Terminal\" to do script \"cd $LIVESTREAM_DIR && conda activate $CONDA_ENV && python send_questions.py $LECTURE_NUMBER\"" &
SEND_QUESTIONS_PID=$!

echo "Processes will terminate after $((DURATION / 60)) minutes..."
sleep $DURATION

echo "Time's up! Terminating all processes..."
kill $NODE_MEDIA_SERVER_PID $FFMPEG_PID $APP2_PID $SEND_QUESTIONS_PID 2>/dev/null

echo "All processes terminated."
