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
  id 1708
  arrival_time 37878.61311456827
  lifetime 1815.4971556548735
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 38
    gpu 33
    rom 39
  ]
  node [
    id 1
    label "1"
    cpu 25
    gpu 13
    rom 27
  ]
  node [
    id 2
    label "2"
    cpu 49
    gpu 18
    rom 18
  ]
  node [
    id 3
    label "3"
    cpu 45
    gpu 40
    rom 47
  ]
  node [
    id 4
    label "4"
    cpu 45
    gpu 36
    rom 0
  ]
  node [
    id 5
    label "5"
    cpu 47
    gpu 11
    rom 23
  ]
  node [
    id 6
    label "6"
    cpu 45
    gpu 1
    rom 31
  ]
  node [
    id 7
    label "7"
    cpu 38
    gpu 3
    rom 15
  ]
  node [
    id 8
    label "8"
    cpu 38
    gpu 34
    rom 14
  ]
  node [
    id 9
    label "9"
    cpu 21
    gpu 44
    rom 28
  ]
  node [
    id 10
    label "10"
    cpu 43
    gpu 36
    rom 16
  ]
  node [
    id 11
    label "11"
    cpu 17
    gpu 36
    rom 17
  ]
  node [
    id 12
    label "12"
    cpu 50
    gpu 8
    rom 32
  ]
  node [
    id 13
    label "13"
    cpu 47
    gpu 0
    rom 0
  ]
  edge [
    source 0
    target 1
    bw 19
  ]
  edge [
    source 1
    target 2
    bw 5
  ]
  edge [
    source 2
    target 3
    bw 17
  ]
  edge [
    source 3
    target 4
    bw 35
  ]
  edge [
    source 4
    target 5
    bw 30
  ]
  edge [
    source 5
    target 6
    bw 44
  ]
  edge [
    source 6
    target 7
    bw 33
  ]
  edge [
    source 7
    target 8
    bw 19
  ]
  edge [
    source 8
    target 9
    bw 44
  ]
  edge [
    source 9
    target 10
    bw 22
  ]
  edge [
    source 10
    target 11
    bw 0
  ]
  edge [
    source 11
    target 12
    bw 30
  ]
  edge [
    source 12
    target 13
    bw 22
  ]
]
