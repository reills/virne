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
  id 1326
  arrival_time 27893.279353323363
  lifetime 1264.5168049496735
  num_nodes 3
  type "path"
  node [
    id 0
    label "0"
    cpu 41
    gpu 43
    rom 32
  ]
  node [
    id 1
    label "1"
    cpu 18
    gpu 28
    rom 18
  ]
  node [
    id 2
    label "2"
    cpu 46
    gpu 26
    rom 40
  ]
  edge [
    source 0
    target 1
    bw 46
  ]
  edge [
    source 1
    target 2
    bw 4
  ]
]
