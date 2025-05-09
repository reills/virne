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
  id 619
  arrival_time 12808.882319104381
  lifetime 421.2724605842356
  num_nodes 9
  type "path"
  node [
    id 0
    label "0"
    cpu 37
    gpu 2
    rom 1
  ]
  node [
    id 1
    label "1"
    cpu 40
    gpu 19
    rom 13
  ]
  node [
    id 2
    label "2"
    cpu 10
    gpu 8
    rom 32
  ]
  node [
    id 3
    label "3"
    cpu 49
    gpu 22
    rom 21
  ]
  node [
    id 4
    label "4"
    cpu 12
    gpu 33
    rom 28
  ]
  node [
    id 5
    label "5"
    cpu 32
    gpu 17
    rom 16
  ]
  node [
    id 6
    label "6"
    cpu 45
    gpu 49
    rom 34
  ]
  node [
    id 7
    label "7"
    cpu 47
    gpu 16
    rom 3
  ]
  node [
    id 8
    label "8"
    cpu 7
    gpu 23
    rom 23
  ]
  edge [
    source 0
    target 1
    bw 40
  ]
  edge [
    source 1
    target 2
    bw 2
  ]
  edge [
    source 2
    target 3
    bw 32
  ]
  edge [
    source 3
    target 4
    bw 5
  ]
  edge [
    source 4
    target 5
    bw 47
  ]
  edge [
    source 5
    target 6
    bw 19
  ]
  edge [
    source 6
    target 7
    bw 43
  ]
  edge [
    source 7
    target 8
    bw 32
  ]
]
