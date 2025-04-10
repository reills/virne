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
  id 815
  arrival_time 16843.533825292434
  lifetime 258.3104721703954
  num_nodes 9
  type "path"
  node [
    id 0
    label "0"
    cpu 25
    gpu 11
    rom 40
  ]
  node [
    id 1
    label "1"
    cpu 43
    gpu 34
    rom 19
  ]
  node [
    id 2
    label "2"
    cpu 44
    gpu 21
    rom 48
  ]
  node [
    id 3
    label "3"
    cpu 7
    gpu 5
    rom 19
  ]
  node [
    id 4
    label "4"
    cpu 12
    gpu 28
    rom 43
  ]
  node [
    id 5
    label "5"
    cpu 8
    gpu 47
    rom 9
  ]
  node [
    id 6
    label "6"
    cpu 42
    gpu 14
    rom 26
  ]
  node [
    id 7
    label "7"
    cpu 10
    gpu 19
    rom 50
  ]
  node [
    id 8
    label "8"
    cpu 11
    gpu 11
    rom 11
  ]
  edge [
    source 0
    target 1
    bw 21
  ]
  edge [
    source 1
    target 2
    bw 35
  ]
  edge [
    source 2
    target 3
    bw 32
  ]
  edge [
    source 3
    target 4
    bw 41
  ]
  edge [
    source 4
    target 5
    bw 30
  ]
  edge [
    source 5
    target 6
    bw 36
  ]
  edge [
    source 6
    target 7
    bw 41
  ]
  edge [
    source 7
    target 8
    bw 36
  ]
]
