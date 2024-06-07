from frechet_audio_distance import FrechetAudioDistance

# to use `vggish`
frechet = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=True
)
# fad_score = frechet.score("/path/to/reference/set", "/path/to/generation/set", dtype="float32")
fad_score = frechet.score("data/slakh2100/fad/background", "output/partial_generation/BDG", dype="float32")
print(fad_score)