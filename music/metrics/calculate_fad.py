from frechet_audio_distance import FrechetAudioDistance

# to use `vggish`
frechet = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=True
)
fad_score = frechet.score("/path/to/gt/set", "/path/to/generation/set")
print(fad_score)