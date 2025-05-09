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
  id 1220
  arrival_time 25299.151688985286
  lifetime 4961.135439164469
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 1
    gpu 40
    rom 44
  ]
  node [
    id 1
    label "1"
    cpu 20
    gpu 33
    rom 34
  ]
  node [
    id 2
    label "2"
    cpu 2
    gpu 5
    rom 3
  ]
  node [
    id 3
    label "3"
    cpu 2
    gpu 41
    rom 30
  ]
  node [
    id 4
    label "4"
    cpu 42
    gpu 9
    rom 31
  ]
  node [
    id 5
    label "5"
    cpu 34
    gpu 14
    rom 22
  ]
  node [
    id 6
    label "6"
    cpu 31
    gpu 10
    rom 13
  ]
  node [
    id 7
    label "7"
    cpu 38
    gpu 24
    rom 12
  ]
  node [
    id 8
    label "8"
    cpu 14
    gpu 21
    rom 23
  ]
  node [
    id 9
    label "9"
    cpu 46
    gpu 34
    rom 13
  ]
  edge [
    source 0
    target 1
    bw 34
  ]
  edge [
    source 1
    target 2
    bw 26
  ]
  edge [
    source 2
    target 3
    bw 12
  ]
  edge [
    source 3
    target 4
    bw 44
  ]
  edge [
    source 4
    target 5
    bw 3
  ]
  edge [
    source 5
    target 6
    bw 8
  ]
  edge [
    source 6
    target 7
    bw 25
  ]
  edge [
    source 7
    target 8
    bw 42
  ]
  edge [
    source 8
    target 9
    bw 48
  ]
]
