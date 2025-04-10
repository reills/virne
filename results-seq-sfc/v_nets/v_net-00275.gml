graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 275
  arrival_time 5298.079537398309
  lifetime 455.9623737977693
  num_nodes 2
  type "path"
  node [
    id 0
    label "0"
    cpu 13
    gpu 22
    rom 38
  ]
  node [
    id 1
    label "1"
    cpu 28
    gpu 50
    rom 0
  ]
  edge [
    source 0
    target 1
    bw 27
  ]
]
