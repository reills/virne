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
  id 1572
  arrival_time 35066.15173438858
  lifetime 1906.103837716576
  num_nodes 15
  type "path"
  node [
    id 0
    label "0"
    cpu 33
    gpu 15
    rom 36
  ]
  node [
    id 1
    label "1"
    cpu 11
    gpu 49
    rom 50
  ]
  node [
    id 2
    label "2"
    cpu 44
    gpu 34
    rom 10
  ]
  node [
    id 3
    label "3"
    cpu 26
    gpu 45
    rom 25
  ]
  node [
    id 4
    label "4"
    cpu 40
    gpu 23
    rom 44
  ]
  node [
    id 5
    label "5"
    cpu 1
    gpu 33
    rom 2
  ]
  node [
    id 6
    label "6"
    cpu 0
    gpu 38
    rom 8
  ]
  node [
    id 7
    label "7"
    cpu 4
    gpu 20
    rom 17
  ]
  node [
    id 8
    label "8"
    cpu 46
    gpu 6
    rom 50
  ]
  node [
    id 9
    label "9"
    cpu 36
    gpu 13
    rom 23
  ]
  node [
    id 10
    label "10"
    cpu 49
    gpu 12
    rom 0
  ]
  node [
    id 11
    label "11"
    cpu 17
    gpu 31
    rom 23
  ]
  node [
    id 12
    label "12"
    cpu 37
    gpu 10
    rom 38
  ]
  node [
    id 13
    label "13"
    cpu 22
    gpu 12
    rom 47
  ]
  node [
    id 14
    label "14"
    cpu 37
    gpu 30
    rom 15
  ]
  edge [
    source 0
    target 1
    bw 32
  ]
  edge [
    source 1
    target 2
    bw 6
  ]
  edge [
    source 2
    target 3
    bw 38
  ]
  edge [
    source 3
    target 4
    bw 40
  ]
  edge [
    source 4
    target 5
    bw 39
  ]
  edge [
    source 5
    target 6
    bw 11
  ]
  edge [
    source 6
    target 7
    bw 45
  ]
  edge [
    source 7
    target 8
    bw 28
  ]
  edge [
    source 8
    target 9
    bw 44
  ]
  edge [
    source 9
    target 10
    bw 46
  ]
  edge [
    source 10
    target 11
    bw 34
  ]
  edge [
    source 11
    target 12
    bw 12
  ]
  edge [
    source 12
    target 13
    bw 11
  ]
  edge [
    source 13
    target 14
    bw 6
  ]
]
