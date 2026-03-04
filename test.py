from faster_whisper import WhisperModel

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

# segments, info = model.transcribe("/mnt/d/order.wav", beam_size=5, initial_prompt="Words spoken:모닝 브리즈, 넛츠 카라멜 드림, 미드나잇 모카, 허니 클라우드 라떼, 바닐라 포레스트, 시트러스 에스프레소 스파클, 로얄 밀크 커피,  아몬드 크런치 마키아또,  블루베리 콜드브루 피즈, 솔트 카라멜 애프터눈")
segments, info = model.transcribe("/mnt/d/order.wav", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))