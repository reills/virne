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
  id 1564
  arrival_time 34997.8940636244
  lifetime 47.57272190511899
  num_nodes 6
  type "path"
  node [
    id 0
    label "0"
    cpu 36
    gpu 49
    rom 44
  ]
  node [
    id 1
    label "1"
    cpu 0
    gpu 12
    rom 33
  ]
  node [
    id 2
    label "2"
    cpu 12
    gpu 16
    rom 45
  ]
  node [
    id 3
    label "3"
    cpu 24
    gpu 30
    rom 8
  ]
  node [
    id 4
    label "4"
    cpu 28
    gpu 43
    rom 28
  ]
  node [
    id 5
    label "5"
    cpu 32
    gpu 47
    rom 17
  ]
  edge [
    source 0
    target 1
    bw 21
  ]
  edge [
    source 1
    target 2
    bw 34
  ]
  edge [
    source 2
    target 3
    bw 40
  ]
  edge [
    source 3
    target 4
    bw 42
  ]
  edge [
    source 4
    target 5
    bw 41
  ]
]
