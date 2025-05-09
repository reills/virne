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
  id 851
  arrival_time 17577.506180111784
  lifetime 2793.336482335647
  num_nodes 4
  type "path"
  node [
    id 0
    label "0"
    cpu 26
    gpu 5
    rom 3
  ]
  node [
    id 1
    label "1"
    cpu 28
    gpu 31
    rom 3
  ]
  node [
    id 2
    label "2"
    cpu 22
    gpu 13
    rom 18
  ]
  node [
    id 3
    label "3"
    cpu 43
    gpu 23
    rom 9
  ]
  edge [
    source 0
    target 1
    bw 0
  ]
  edge [
    source 1
    target 2
    bw 36
  ]
  edge [
    source 2
    target 3
    bw 10
  ]
]
