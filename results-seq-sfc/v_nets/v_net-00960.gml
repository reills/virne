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
  id 960
  arrival_time 20391.923070981044
  lifetime 98.20156369945435
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 10
    gpu 33
    rom 48
  ]
  node [
    id 1
    label "1"
    cpu 0
    gpu 7
    rom 20
  ]
  node [
    id 2
    label "2"
    cpu 33
    gpu 47
    rom 5
  ]
  node [
    id 3
    label "3"
    cpu 23
    gpu 0
    rom 47
  ]
  node [
    id 4
    label "4"
    cpu 11
    gpu 5
    rom 12
  ]
  node [
    id 5
    label "5"
    cpu 43
    gpu 31
    rom 9
  ]
  node [
    id 6
    label "6"
    cpu 9
    gpu 46
    rom 23
  ]
  node [
    id 7
    label "7"
    cpu 49
    gpu 7
    rom 41
  ]
  node [
    id 8
    label "8"
    cpu 20
    gpu 35
    rom 5
  ]
  node [
    id 9
    label "9"
    cpu 38
    gpu 50
    rom 37
  ]
  edge [
    source 0
    target 1
    bw 3
  ]
  edge [
    source 1
    target 2
    bw 41
  ]
  edge [
    source 2
    target 3
    bw 23
  ]
  edge [
    source 3
    target 4
    bw 23
  ]
  edge [
    source 4
    target 5
    bw 35
  ]
  edge [
    source 5
    target 6
    bw 26
  ]
  edge [
    source 6
    target 7
    bw 37
  ]
  edge [
    source 7
    target 8
    bw 0
  ]
  edge [
    source 8
    target 9
    bw 47
  ]
]
