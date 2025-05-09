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
  id 1221
  arrival_time 25331.34124154557
  lifetime 2851.6354792696393
  num_nodes 3
  type "path"
  node [
    id 0
    label "0"
    cpu 33
    gpu 31
    rom 50
  ]
  node [
    id 1
    label "1"
    cpu 11
    gpu 1
    rom 22
  ]
  node [
    id 2
    label "2"
    cpu 26
    gpu 34
    rom 37
  ]
  edge [
    source 0
    target 1
    bw 2
  ]
  edge [
    source 1
    target 2
    bw 43
  ]
]
