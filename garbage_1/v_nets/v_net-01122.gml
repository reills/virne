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
  id 1122
  arrival_time 23337.36800982204
  lifetime 1096.7212870541741
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 29
    gpu 22
    rom 1
  ]
  node [
    id 1
    label "1"
    cpu 46
    gpu 26
    rom 23
  ]
  node [
    id 2
    label "2"
    cpu 44
    gpu 25
    rom 20
  ]
  node [
    id 3
    label "3"
    cpu 28
    gpu 27
    rom 44
  ]
  node [
    id 4
    label "4"
    cpu 10
    gpu 18
    rom 1
  ]
  node [
    id 5
    label "5"
    cpu 16
    gpu 31
    rom 13
  ]
  node [
    id 6
    label "6"
    cpu 24
    gpu 50
    rom 8
  ]
  node [
    id 7
    label "7"
    cpu 40
    gpu 37
    rom 20
  ]
  node [
    id 8
    label "8"
    cpu 35
    gpu 49
    rom 1
  ]
  node [
    id 9
    label "9"
    cpu 34
    gpu 0
    rom 47
  ]
  node [
    id 10
    label "10"
    cpu 26
    gpu 23
    rom 33
  ]
  edge [
    source 0
    target 1
    bw 4
  ]
  edge [
    source 1
    target 2
    bw 21
  ]
  edge [
    source 2
    target 3
    bw 25
  ]
  edge [
    source 3
    target 4
    bw 23
  ]
  edge [
    source 4
    target 5
    bw 22
  ]
  edge [
    source 5
    target 6
    bw 36
  ]
  edge [
    source 6
    target 7
    bw 36
  ]
  edge [
    source 7
    target 8
    bw 32
  ]
  edge [
    source 8
    target 9
    bw 46
  ]
  edge [
    source 9
    target 10
    bw 0
  ]
]
