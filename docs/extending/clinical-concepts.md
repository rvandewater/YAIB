# Clinical concepts

YAIB's cohorts are built from clinical concepts (heart rate, creatinine, vasopressors, ...) that
[ricu](https://github.com/eth-mds/ricu) defines for every supported dataset. New tasks often need more variables;
medications are a common example.

For many concepts, all that is needed is a JSON entry that lists the measurement IDs of the concept in each dataset.
Using potassium chloride, a fluid often given in the ICU, as an example, the entry describes where potassium chloride
administrations are recorded in MIMIC, eICU, HiRID and AUMCdb. You can then:

- append the entry to ricu's concept dictionary, for additions that should become part of ricu, or
- keep it in your own folder and add that folder to the search path of `load_dictionary` with its `cfg_dirs`
    argument.

Complex transformations, such as calculating hourly rates, are supported by ricu's helper functions. If a concept needs
a custom calculation, provide it as a function through the `callback` element.

See the [ricu documentation](https://eth-mds.github.io/ricu/) for the concept format, and
[YAIB-cohorts](https://github.com/rvandewater/YAIB-cohorts) (its `ricu-extensions` folder) for the concepts YAIB adds.
