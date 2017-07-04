- ~~n-grams for text~~ (added)
- stemming for text
- Use more NLP techniques word2vec, glove and others
- ~~find information about the target classes to determine what useful
 features may be created~~ We now know that the output classes are
 9 mutation effects - https://www.kaggle.com/c/msk-redefining-cancer-treatment/discussion/35336
- ~~find resources explaining what the `Gene` code means for potential feature
 ideas~~ (implented)
- find resources explaining what the `Variation` code means for potential
 feature ideas. `Variation` really needs some work - it has 3k unique values out of 5k.
 Some values do occur frequently but the field still has a huge tail:

|Value|Count|
|-----|-----|
|Truncating Mutations    |93|
|Deletion                |74|
|Amplification           |71|
|Fusions                 |34|
|Overexpression          | 6|
|G12V                    | 4|
|E17K                    | 3|

- ~~Pick a model~~ going to stick with MS LitghGBM
- Think about features selection
- Extend gene features by joining on specialist data-sets via the
ids specified in the http://www.genenames.org/cgi-bin/genefamilies dataset.