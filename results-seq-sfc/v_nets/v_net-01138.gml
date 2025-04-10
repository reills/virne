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
  id 1138
  arrival_time 23846.170018014906
  lifetime 164.34762163678027
  num_nodes 6
  type "path"
  node [
    id 0
    label "0"
    cpu 8
    gpu 38
    rom 2
  ]
  node [
    id 1
    label "1"
    cpu 35
    gpu 37
    rom 43
  ]
  node [
    id 2
    label "2"
    cpu 9
    gpu 47
    rom 30
  ]
  node [
    id 3
    label "3"
    cpu 23
    gpu 42
    rom 26
  ]
  node [
    id 4
    label "4"
    cpu 39
    gpu 49
    rom 14
  ]
  node [
    id 5
    label "5"
    cpu 0
    gpu 38
    rom 40
  ]
  edge [
    source 0
    target 1
    bw 25
  ]
  edge [
    source 1
    target 2
    bw 14
  ]
  edge [
    source 2
    target 3
    bw 28
  ]
  edge [
    source 3
    target 4
    bw 24
  ]
  edge [
    source 4
    target 5
    bw 46
  ]
]
