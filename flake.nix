{
  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs/nixos-unstable";
      #url = "github:DerDennisOP/nixpkgs/wikipedia2vec";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
  };
  outputs = { nixpkgs, flake-utils, ... }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };
    in rec {
      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [
          (python3.withPackages(ps: with ps; [
            ipython jupyter spyder qtconsole
            numpy matplotlib
            pandas plotly ipywidgets notebook
            scipy keras #tensorflow dm-tree
            torch transformers accelerate bitsandbytes torchvision evaluate jiwer tiktoken
            torchinfo wandb tqdm
            datasets kaggle
            scikit-image urllib3 scikit-learn
            opencv4
            sympy
            joblib marisa-trie
            rdflib
            pytesseract
          ]))
        ];
        shellHook = ''
            export PYTHONPATH="$PYTHON_PATH:`pwd`/src"
            #jupyter notebook
            #jupyter lab
            #spyder
            #exit
        '';
      };
      devShells.${system} = {
        doc = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [ mdbook mdbook-mermaid ];
          shellHook = ''
            cd doc
            mdbook-mermaid install
            mdbook serve
          '';
        };
        get_corpus = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            lynx
            #poppler_utils wget
          ];
          shellHook = ''
            mkdir example
            wget https://www.gutenberg.org/cache/epub/2229/pg2229.txt -O example/corpus.txt
            #https://github.com/google-research-datasets/natural-questions
            #https://huggingface.co/datasets/wiki_qa
            #https://openwebtext2.readthedocs.io/en/latest/
            #https://info.arxiv.org/help/bulk_data.html
          '';
        };
      };
    }
  );
}
