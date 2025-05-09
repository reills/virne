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
  id 1561
  arrival_time 34939.69869790372
  lifetime 1396.8131797092992
  num_nodes 5
  type "path"
  node [
    id 0
    label "0"
    cpu 31
    gpu 0
    rom 2
  ]
  node [
    id 1
    label "1"
    cpu 37
    gpu 40
    rom 13
  ]
  node [
    id 2
    label "2"
    cpu 20
    gpu 1
    rom 13
  ]
  node [
    id 3
    label "3"
    cpu 45
    gpu 46
    rom 43
  ]
  node [
    id 4
    label "4"
    cpu 8
    gpu 27
    rom 43
  ]
  edge [
    source 0
    target 1
    bw 39
  ]
  edge [
    source 1
    target 2
    bw 48
  ]
  edge [
    source 2
    target 3
    bw 46
  ]
  edge [
    source 3
    target 4
    bw 50
  ]
]
