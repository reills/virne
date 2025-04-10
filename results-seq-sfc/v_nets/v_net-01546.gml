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
  id 1546
  arrival_time 34164.1481594381
  lifetime 355.2885088853454
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 5
    gpu 16
    rom 25
  ]
  node [
    id 1
    label "1"
    cpu 5
    gpu 33
    rom 39
  ]
  node [
    id 2
    label "2"
    cpu 47
    gpu 12
    rom 27
  ]
  node [
    id 3
    label "3"
    cpu 45
    gpu 47
    rom 49
  ]
  node [
    id 4
    label "4"
    cpu 26
    gpu 9
    rom 29
  ]
  node [
    id 5
    label "5"
    cpu 14
    gpu 23
    rom 33
  ]
  node [
    id 6
    label "6"
    cpu 22
    gpu 5
    rom 1
  ]
  node [
    id 7
    label "7"
    cpu 40
    gpu 5
    rom 10
  ]
  node [
    id 8
    label "8"
    cpu 49
    gpu 44
    rom 16
  ]
  node [
    id 9
    label "9"
    cpu 36
    gpu 27
    rom 8
  ]
  node [
    id 10
    label "10"
    cpu 11
    gpu 44
    rom 46
  ]
  node [
    id 11
    label "11"
    cpu 48
    gpu 46
    rom 42
  ]
  node [
    id 12
    label "12"
    cpu 36
    gpu 28
    rom 33
  ]
  node [
    id 13
    label "13"
    cpu 5
    gpu 19
    rom 4
  ]
  edge [
    source 0
    target 1
    bw 35
  ]
  edge [
    source 1
    target 2
    bw 14
  ]
  edge [
    source 2
    target 3
    bw 49
  ]
  edge [
    source 3
    target 4
    bw 18
  ]
  edge [
    source 4
    target 5
    bw 33
  ]
  edge [
    source 5
    target 6
    bw 37
  ]
  edge [
    source 6
    target 7
    bw 24
  ]
  edge [
    source 7
    target 8
    bw 39
  ]
  edge [
    source 8
    target 9
    bw 1
  ]
  edge [
    source 9
    target 10
    bw 15
  ]
  edge [
    source 10
    target 11
    bw 16
  ]
  edge [
    source 11
    target 12
    bw 5
  ]
  edge [
    source 12
    target 13
    bw 0
  ]
]
