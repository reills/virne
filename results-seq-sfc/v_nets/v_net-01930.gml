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
  id 1930
  arrival_time 42286.81511947728
  lifetime 1065.9389143074607
  num_nodes 6
  type "path"
  node [
    id 0
    label "0"
    cpu 44
    gpu 37
    rom 36
  ]
  node [
    id 1
    label "1"
    cpu 40
    gpu 9
    rom 10
  ]
  node [
    id 2
    label "2"
    cpu 44
    gpu 2
    rom 39
  ]
  node [
    id 3
    label "3"
    cpu 34
    gpu 33
    rom 20
  ]
  node [
    id 4
    label "4"
    cpu 25
    gpu 1
    rom 16
  ]
  node [
    id 5
    label "5"
    cpu 43
    gpu 10
    rom 4
  ]
  edge [
    source 0
    target 1
    bw 5
  ]
  edge [
    source 1
    target 2
    bw 49
  ]
  edge [
    source 2
    target 3
    bw 43
  ]
  edge [
    source 3
    target 4
    bw 27
  ]
  edge [
    source 4
    target 5
    bw 20
  ]
]
