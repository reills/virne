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
  id 1384
  arrival_time 29249.49933314629
  lifetime 312.11810897987186
  num_nodes 3
  type "path"
  node [
    id 0
    label "0"
    cpu 30
    gpu 4
    rom 6
  ]
  node [
    id 1
    label "1"
    cpu 47
    gpu 24
    rom 26
  ]
  node [
    id 2
    label "2"
    cpu 8
    gpu 39
    rom 41
  ]
  edge [
    source 0
    target 1
    bw 47
  ]
  edge [
    source 1
    target 2
    bw 32
  ]
]
