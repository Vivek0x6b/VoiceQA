# Test speech: source and processing

`magi_8k.wav` is an 18.1 s passage from a LibriVox recording, used by the
real-model STT tests and by `voiceqa bench stt`. `utterances.json` holds the
start, end and reference text of each of its four utterances.

## Source

- Book: The Gift of the Magi, by O. Henry (first published 1905, in book form 1906).
- Chapter: the whole story is one section. The passage starts at about 3:14.9 in the recording.
- Reader: Betsie Bush (LibriVox reader 32), recorded December 2005.
- LibriVox page: https://librivox.org/the-gift-of-the-magi-by-o-henry/
- Audio file: https://www.archive.org/download/giftofmagi/gift_of_the_magi_henry_blb_64kb.mp3
- Text: Project Gutenberg eBook #7256, https://www.gutenberg.org/ebooks/7256 (the text source LibriVox lists for this recording).

## Public domain

All LibriVox recordings are in the public domain, as the recording itself
says in its opening. O. Henry died in 1910 and the story was published in
1905, so the text is in the public domain in the United States and in
countries where copyright runs for the author's life plus 70 years or less.

## Reference text

The reference text is the book's own text for the passage, copied from the
Project Gutenberg edition, not a speech recogniser's output:

> Della, being slender, had mastered the art.
>
> Suddenly she whirled from the window and stood before the glass. Her
> eyes were shining brilliantly, but her face had lost its color within
> twenty seconds. Rapidly she pulled down her hair and let it fall to its
> full length.

The passage was located in the recording with a local faster-whisper
`base.en` run over the first four minutes. Both `tiny.en` and `base.en` hear
the first words as "Stella, the in slender". The reading was not checked by
ear, so the reader may say something slightly different there; the tests
allow for it.

## Processing

1. Downloaded the 64 kbps MP3 above (24 kHz, stereo).
2. Decoded it to float samples with PyAV, averaged the two channels and
   converted to int16.
3. Cut 193.0 s to 215.0 s of the MP3 and resampled it to 8 kHz with
   `voiceqa.audio.dsp.resample`.
4. Trimmed that to 194.9 s to 213.0 s of the MP3 (18.1 s), which starts and
   ends in a pause.
5. Saved it as 8 kHz mono 16-bit PCM WAV.
6. Found utterance boundaries by energy: 20 ms frames, RMS in dBFS, a frame
   counts as speech when it is above the 95th percentile level minus 35 dB,
   and a quiet run of at least 440 ms ends an utterance. Spans shorter than
   200 ms are dropped. Times are rounded to the 20 ms frame grid.
7. Checked the four boundaries against the `base.en` segment times for the
   same passage. They agree within about 0.6 s, and each detected
   utterance is one of the book's sentences.
