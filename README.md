# markov

A simple text generator using markov chains. Made for fun :)

```
Usage: markov [OPTIONS] [INPUT_FILE] [INITIAL_PHRASE]

Arguments:
  [INPUT_FILE]
          A text file containing text to learn off of, or a .bin file containing a previously saved transition matrix

  [INITIAL_PHRASE]
          An initial text phrase to start predicting off of. Markov chains are memoryless so only
        the final few tokens will be used.

Options:
  -s, --output-size <OUTPUT_SIZE>
          Number of tokens to generate
        
          [default: 200]

  -t, --state-size <STATE_SIZE>
          The number of tokens to use per state in the transition matrix.
          
          When loading an existing matrix this option is ignored.
          
          [default: 2]

      --save[=<SAVE>]
          Save transition matrix to a binary file `SAVE` instead of generating tokens

  -h, --help
          Print help (see a summary with '-h')

```

## Example Usage:
```bash
$ markov hamlet.txt "Hi homer" --output-size=50
 Hi Homer To offer, To not, if not, if England, from England news from the
news's the What's! What guards! heavenly guards You heavenly, You me, Believe
me. Believe Guil.? Guil toil? a toil into a me into

$ markov hamlet.txt --state-size=3 --save=hamlet.markov
$ markov hamlet.markov "Hi homer" --output-size=50
 Hi Homer tedious old fools These tedious old. These tedious Ham. These. Ham.
weasel. Ham a weasel. like a weasel'd like a back'd like is back'd It is back.
It is music. It healthful music. as healthful music
```
