# TEXT2SQL_RL_SPIDER

## Setup data

This project expects the Spider dataset at `Source/spider_data`, which matches
the default path in `spider_env.load_spider_samples()`.

Install the downloader once on a new box:

```sh
uv pip install gdown
```

Then fetch and verify the dataset:

```sh
make data
```

The Makefile downloads the official Spider 1.0 `spider_data.zip`, unzips it into
`Source/`, and checks for the train/dev JSON files plus the database directory.
The data files are ignored by git.

### Sources 
@inproceedings{Yu&al.18c,
  year =         2018,
  title =        {Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  booktitle =    {EMNLP},
  author =       {Tao Yu and Rui Zhang and Kai Yang and Michihiro Yasunaga and Dongxu Wang and Zifan Li and James Ma and Irene Li and Qingning Yao and Shanelle Roman and Zilin Zhang and Dragomir Radev }
}
