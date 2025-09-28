import assemblyai as aai

aai.settings.api_key = "3aecc3231d964f61b37e334f18fbee0c"

audio_file = "2-assemblyai/AUDIO/Dollar Drop Prediction.mp3"

config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)

print("🔄 Transcription in progress...")
transcript = aai.Transcriber(config=config).transcribe(audio_file)

if transcript.status == "completed":
  print("✅ Transcription completed.")
  
if transcript.status == "error":
  raise RuntimeError(f"Transcription failed: {transcript.error}")


print(transcript.text)