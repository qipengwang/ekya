# Frequently Asked Questions

1. When installing ray with `pip install -e . --verbose` and encountering the 
   error `"[ray] [bazel] build failure, error --experimental_ui_deduplicate
   unrecognized"`. 

    Please checkout this
    [issue](https://github.com/ray-project/ray/issues/11237). If other versions
    of `bazel` are installed, please install `bazel-3.2.0` following instructions
    from
    [here](https://docs.bazel.build/versions/main/install-compile-source.html)
    and compile ray useing `bazel-3.2.0`.

