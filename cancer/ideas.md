- ~~n-grams for text~~ (added)
- stemming for text
- find information about the target classes to determine what useful
 features may be created
- find resources explaining what the `Gene` code means for potential feature
 ideas
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

- Pick a model
- Think about features selection
- Gridsearch preserving the class-prior distribution?