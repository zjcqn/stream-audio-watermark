An implementation of live-streaming audio watermark based on [AudioSeal](https://github.com/facebookresearch/audioseal) and [Demucs](https://github.com/facebookresearch/denoiser).

Drawing on the framework of Audioseal, the system has achieved voice digital watermarking at the sample point level. Most crucially, it is adapted for streaming live broadcast scenarios, capable of injecting watermarks with a delay of just over 10 milliseconds. After parameter tuning and testing, our version converges more readily than Audioseal.




## References

- Defossez, Alexandre, et al. "Real Time Speech Enhancement in the Waveform Domain." Interspeech, 2020.
- San Roman, Robin, et al. "Proactive Detection of Voice Cloning with Localized Watermarking." ICML, 2024.