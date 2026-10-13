# Data

The panoramic X-rays used by this project are patient images and are not stored in this
repository. Place them here as an `ImageFolder` tree:

```
data/
└── OPG With landmarks/
    ├── female/   01.png, 01_aug_0003.png, ...
    └── male/     02.png, 02_aug_0001.png, ...
```

Pre-augmented copies must keep the source X-ray's name as a prefix followed by `_aug_NNNN`
(repeatable), because `splits.py` uses that prefix to keep every copy of an X-ray in the same
split.

## Before releasing any images

To publish the data, or a sample of it, record:

- **Source**: where the X-rays came from, and who collected them.
- **Consent / ethics**: the approval or consent that covers public release.
- **De-identification**: confirmation that no names, IDs or dates are burned into the images
  or kept in their metadata.
- **Licence**: the terms under which others may use them.
